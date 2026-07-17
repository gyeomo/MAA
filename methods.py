import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def _normalize_weights(w: Optional[torch.Tensor], n: int, device: torch.device) -> torch.Tensor:
    if n <= 0:
        return torch.empty((0,), device=device)
    if w is None:
        return torch.full((n,), 1.0 / n, device=device)
    w = w.clamp_min(0.0)
    s = w.sum()
    if s.item() <= 0:
        return torch.full((n,), 1.0 / n, device=device)
    return w / (s + 1e-8)


def _effective_n(w: torch.Tensor) -> torch.Tensor:
    w = w.clamp_min(0.0)
    sw = w.sum()
    sw2 = (w * w).sum().clamp_min(1e-8)
    return (sw * sw) / sw2


# -------------------------
# Alignment losses (MMD / CORAL / DANN)
# -------------------------
class MMDLoss(nn.Module):
    def __init__(self, sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)):
        super().__init__()
        self.sigmas = tuple(float(s) for s in sigmas)

    @staticmethod
    def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x, y, p=2.0).pow(2)

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d2 = self._pairwise_sq_dists(x, y)
        k = 0.0
        for s in self.sigmas:
            k = k + torch.exp(-d2 / (2.0 * (s ** 2)))
        return k

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if x_s.numel() == 0 or x_t.numel() == 0:
            return x_s.new_tensor(0.0)

        ns, nt = x_s.shape[0], x_t.shape[0]
        device = x_s.device
        ws = _normalize_weights(w_s, ns, device)
        wt = _normalize_weights(w_t, nt, device)

        K_ss = self._rbf_kernel(x_s, x_s)
        K_tt = self._rbf_kernel(x_t, x_t)
        K_st = self._rbf_kernel(x_s, x_t)

        m_ss = (ws[:, None] * ws[None, :] * K_ss).sum()
        m_tt = (wt[:, None] * wt[None, :] * K_tt).sum()
        m_st = (ws[:, None] * wt[None, :] * K_st).sum()

        return (m_ss + m_tt - 2.0 * m_st).clamp_min(0.0)


class CORALLoss(nn.Module):
    def __init__(self, scale: bool = True, with_mean: bool = True, min_eff_n: float = 2.0):
        super().__init__()
        self.scale = bool(scale)
        self.with_mean = bool(with_mean)
        self.min_eff_n = float(min_eff_n)

    @staticmethod
    def _weighted_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        w = w.clamp_min(0.0)
        return (w[:, None] * x).sum(dim=0) / (w.sum() + 1e-8)

    @staticmethod
    def _weighted_cov(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        w = w.clamp_min(0.0)
        mu = CORALLoss._weighted_mean(x, w)
        xc = x - mu[None, :]
        sw = w.sum().clamp_min(1e-8)
        denom = (sw - (w * w).sum() / sw).clamp_min(1e-8)
        return (xc.t() @ (w[:, None] * xc)) / denom

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if x_s.numel() == 0 or x_t.numel() == 0:
            return x_s.new_tensor(0.0)

        ns, nt = x_s.shape[0], x_t.shape[0]
        device = x_s.device
        ws = _normalize_weights(w_s, ns, device) * float(ns)
        wt = _normalize_weights(w_t, nt, device) * float(nt)

        if _effective_n(ws).item() < self.min_eff_n or _effective_n(wt).item() < self.min_eff_n:
            return x_s.new_tensor(0.0)

        Cs = self._weighted_cov(x_s, ws)
        Ct = self._weighted_cov(x_t, wt)

        d = x_s.shape[1]
        loss = (Cs - Ct).pow(2).mean()

        if self.with_mean:
            mu_s = self._weighted_mean(x_s, ws)
            mu_t = self._weighted_mean(x_t, wt)
            loss = loss + (mu_s - mu_t).pow(2).mean()

        if self.scale:
            loss = loss * d
        return loss


class DANNLoss(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        self.grl_lambda = 1.0

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor, w_s: torch.Tensor, w_t: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_s, z_t], dim=0)
        d = torch.cat([
            torch.zeros(z_s.shape[0], device=z.device, dtype=torch.long),
            torch.ones(z_t.shape[0], device=z.device, dtype=torch.long),
        ], dim=0)

        w = torch.cat([w_s, w_t], dim=0).clamp_min(1e-12)
        w = w / w.sum()

        z_grl = _GradReverse.apply(z, float(self.grl_lambda))
        logits = self.disc(z_grl)
        ce = F.cross_entropy(logits, d, reduction="none")
        return (w * ce).sum()

def _entropy_from_logits(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Entropy per sample (B,) from logits (B,K)."""
    p = F.softmax(logits, dim=-1).clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)

class MAA(nn.Module):
    """
    multiclass version
    - normal(y==0): view weight = uniform
    - defect(y!=0): view score based on p(non-OK|view) + normal-prototype distance
    - weights are applied to the alignment loss only
    """

    def __init__(self, args, embedding_dim: int, num_classes: int):
        super().__init__()
        self.D = int(embedding_dim)
        self.K = int(num_classes)
        if self.K < 2:
            raise ValueError("num_classes must be >= 2.")

        # alignment loss
        self.align_loss_name: str = str(getattr(args, "align_loss", "dann")).lower()
        self.lambda_align = 1.0
        self.multiplier: float = float(getattr(args, "multiplier", 2.0))
        self.eps = 1e-12

        # temperature
        self.tau_orc: float = float(getattr(args, "tau_orc", 0.05))

        # EMA normal prototype
        self.use_ema_proto: bool = bool(getattr(args, "use_ema_proto", True))
        mom = float(getattr(args, "proto_momentum", 0.9))
        self.proto_momentum_s = mom
        self.proto_momentum_t = mom
        self.proto_min_norm: int = int(getattr(args, "proto_min_norm", 1))

        self.register_buffer("mu_norm_s_ema", torch.empty(0))
        self.register_buffer("mu_norm_t_ema", torch.empty(0))
        self.register_buffer("norm_seen_s", torch.zeros((), dtype=torch.long))
        self.register_buffer("norm_seen_t", torch.zeros((), dtype=torch.long))
        self.register_buffer("ema_steps_s", torch.zeros((), dtype=torch.long))
        self.register_buffer("ema_steps_t", torch.zeros((), dtype=torch.long))

        if self.align_loss_name == "mmd":
            self.align = MMDLoss(sigmas=(1.0, 2.0, 4.0, 8.0))
        elif self.align_loss_name == "coral":
            self.align = CORALLoss()
        elif self.align_loss_name == "dann":
            self.align = DANNLoss(feat_dim=self.D, hidden=int(getattr(args, "disc_hidden", 256)))
            self.align.grl_lambda = 1.0
        else:
            raise ValueError("args.align_loss must be one of: 'mmd', 'coral', 'dann'")

        self.opt_disc = None
        if isinstance(self.align, DANNLoss):
            lr_disc = float(getattr(args, "lr_disc", 1e-4))
            wd_disc = float(getattr(args, "wd_disc", 1e-6))
            self.opt_disc = torch.optim.Adam(self.align.disc.parameters(), lr=lr_disc, weight_decay=wd_disc)

    @staticmethod
    def _flatten_views(z: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, V, D = z.shape
        return z.reshape(B * V, D), w.reshape(B * V)

    @staticmethod
    def _zscore_views(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
        return (x - mu) / sd

    @torch.no_grad()
    def _logp_nonok_from_z(self, model, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B,V,D) -> log p(non-OK | view): (B,V)
        OK(class 0) vs non-OK(class 1..K-1)
        """
        B, V, D = z.shape
        zv = z.reshape(B * V, D)
        logits = model.classifier(zv)  # (B*V,K)
        probs = F.softmax(logits, dim=-1)
        p_ok = probs[:, 0]
        p_nonok = (1.0 - p_ok).clamp_min(self.eps)
        logp_nonok = torch.log(p_nonok).reshape(B, V)
        return logp_nonok

    def _align_core(self, z_s: torch.Tensor, z_t: torch.Tensor,
                    w_s: torch.Tensor, w_t: torch.Tensor) -> torch.Tensor:
        if z_s.numel() == 0 or z_t.numel() == 0:
            return z_s.new_tensor(0.0)
        zsf, wsf = self._flatten_views(z_s, w_s)
        ztf, wtf = self._flatten_views(z_t, w_t)
        return self.align(zsf, ztf, wsf, wtf)

    def _align(self, z_s: torch.Tensor, z_t: torch.Tensor,
               w_s: torch.Tensor, w_t: torch.Tensor,
               y_s: Optional[torch.Tensor] = None,
               y_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if (y_s is None) or (y_t is None):
            return self._align_core(z_s, z_t, w_s, w_t)

        total = z_s.new_tensor(0.0)
        denom = z_s.new_tensor(0.0)

        for c in range(self.K):
            ms = (y_s == c)
            mt = (y_t == c)
            if (not ms.any()) or (not mt.any()):
                continue

            weight = self.multiplier if c == 0 else 1.0
            loss_c = weight * self._align_core(z_s[ms], z_t[mt], w_s[ms], w_t[mt])

            alpha = z_s.new_tensor(float(ms.sum().item() + mt.sum().item()))
            total = total + alpha * loss_c
            denom = denom + alpha

        if denom.item() <= 0:
            return z_s.new_tensor(0.0)
        return total / denom.clamp_min(1e-12)

    @torch.no_grad()
    def _update_norm_proto_ema(self, z: torch.Tensor, y: torch.Tensor, domain: str) -> None:
        if (not self.use_ema_proto) or (z.numel() == 0):
            return
        m = (y == 0)
        if not m.any():
            return

        mu_batch = z[m].mean(dim=0).detach()  # (V,D)

        if domain == "s":
            mom = float(self.proto_momentum_s)
            if (self.mu_norm_s_ema.numel() == 0) or (self.mu_norm_s_ema.shape != mu_batch.shape):
                self.mu_norm_s_ema = mu_batch
                self.ema_steps_s = self.ema_steps_s.new_tensor(1, dtype=torch.long)
            else:
                self.mu_norm_s_ema = mom * self.mu_norm_s_ema + (1.0 - mom) * mu_batch
                self.ema_steps_s = self.ema_steps_s + 1
            self.norm_seen_s = self.norm_seen_s + m.sum().to(self.norm_seen_s.dtype)

        elif domain == "t":
            mom = float(self.proto_momentum_t)
            if (self.mu_norm_t_ema.numel() == 0) or (self.mu_norm_t_ema.shape != mu_batch.shape):
                self.mu_norm_t_ema = mu_batch
                self.ema_steps_t = self.ema_steps_t.new_tensor(1, dtype=torch.long)
            else:
                self.mu_norm_t_ema = mom * self.mu_norm_t_ema + (1.0 - mom) * mu_batch
                self.ema_steps_t = self.ema_steps_t + 1
            self.norm_seen_t = self.norm_seen_t + m.sum().to(self.norm_seen_t.dtype)
        else:
            raise ValueError("domain must be 's' or 't'")

    def _get_mu_norm(self, z: torch.Tensor, y: torch.Tensor, domain: str) -> Optional[torch.Tensor]:
        if self.use_ema_proto:
            if domain == "s" and (self.mu_norm_s_ema.numel() != 0) and (int(self.norm_seen_s.item()) >= self.proto_min_norm):
                t = int(self.ema_steps_s.item())
                m = float(self.proto_momentum_s)
                debias = 1.0 - (m ** max(t, 1))
                return self.mu_norm_s_ema / max(debias, 1e-8)
            if domain == "t" and (self.mu_norm_t_ema.numel() != 0) and (int(self.norm_seen_t.item()) >= self.proto_min_norm):
                t = int(self.ema_steps_t.item())
                m = float(self.proto_momentum_t)
                debias = 1.0 - (m ** max(t, 1))
                return self.mu_norm_t_ema / max(debias, 1e-8)

        m_norm = (y == 0)
        if m_norm.any():
            return z[m_norm].mean(dim=0)
        return None

    @torch.no_grad()
    def _normal_logp_prior(self, model, z: torch.Tensor, y: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Per-view log p(non-OK) prior over normal (y==0) samples
        returns:
          mean_v, std_v: (1,V) or (None,None)
        """
        m = (y == 0)
        if (not m.any()) or (z[m].shape[0] < 2):
            return None, None

        logp_norm = self._logp_nonok_from_z(model, z[m])  # (Bn,V)
        mean_v = logp_norm.mean(dim=0, keepdim=True)
        std_v = logp_norm.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        return mean_v, std_v

    @torch.no_grad()
    def _normal_dist_prior(self, z: torch.Tensor, y: torch.Tensor, mu: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Per-view dist(z, mu) prior over normal (y==0) samples
        returns mean_v,std_v: (1,V) or (None,None)
        """
        m = (y == 0)
        if (not m.any()) or (z[m].shape[0] < 2):
            return None, None
        z_norm = z[m]
        dist_norm = ((z_norm - mu[None, :, :]) ** 2).mean(dim=-1)
        mean_v = dist_norm.mean(dim=0, keepdim=True)
        std_v = dist_norm.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        return mean_v, std_v

    @torch.no_grad()
    def _oracle_weights_one_domain(self, model, z: torch.Tensor, y: torch.Tensor, logit: torch.Tensor, domain: str) -> torch.Tensor:
        """
        normal (y==0): uniform
        defect (y!=0): score from p(non-OK|view) + normal-prototype distance
        """
        B, V, D = z.shape
        if B == 0:
            return z.new_empty((0, V))

        w_oracle = z.new_full((B, V), 1.0 / float(V))

        m_def = (y != 0)
        if not m_def.any():
            return w_oracle

        mu_norm = self._get_mu_norm(z, y, domain=domain)
        z_def = z[m_def]

        # (1) defect evidence = log p(non-OK | view)
        logp_nonok = self._logp_nonok_from_z(model, z_def)

        mean_lp, std_lp = self._normal_logp_prior(model, z, y)
        if (mean_lp is None) or (std_lp is None):
            ce_prior = logp_nonok
        else:
            ce_prior = (logp_nonok - mean_lp) / std_lp
        ce_n = self._zscore_views(ce_prior)

        # pooled prediction confidence
        ent_logit = _entropy_from_logits(logit[m_def], eps=self.eps)
        ent_norm_logit = ent_logit / max(1e-8, math.log(float(self.K)))
        conf = (1.0 - ent_norm_logit).clamp(0.0, 1.0).unsqueeze(1)

        # (2) distance term
        if mu_norm is None:
            score = ce_n
        else:
            mu = mu_norm.detach()
            dist = ((z_def - mu[None, :, :]) ** 2).mean(dim=-1)

            mean_d, std_d = self._normal_dist_prior(z, y, mu)
            if (mean_d is None) or (std_d is None):
                dist_prior = dist
            else:
                dist_prior = (dist - mean_d) / std_d
            dist_n = self._zscore_views(dist_prior)

            # keep the original combination scheme
            score = conf * ce_n + dist_n

        w_def = torch.softmax(score / max(self.tau_orc, 1e-6), dim=1)
        w_oracle[m_def] = w_def
        return w_oracle

    def _calcul_weight(self, model, z_s, y_s, z_t, y_t, logit_s, logit_t):
        w_s = self._oracle_weights_one_domain(model, z_s, y_s, logit_s, domain="s")
        w_t = self._oracle_weights_one_domain(model, z_t, y_t, logit_t, domain="t")
        return w_s, w_t

    def train_step(self, model, optimizer, batch_s, batch_t, device, step: int) -> Dict[str, float]:
        model.train()
        self.train()

        x_s, y_s = batch_s
        x_t, y_t = batch_t
        x_s, y_s = x_s.to(device), y_s.to(device)
        x_t, y_t = x_t.to(device), y_t.to(device)

        z_s = model.encode_views(x_s)  # (Bs,V,D)
        z_t = model.encode_views(x_t)  # (Bt,V,D)

        h_s = z_s.mean(dim=1)
        h_t = z_t.mean(dim=1)
        logits_s = model.classifier(h_s)
        logits_t = model.classifier(h_t)

        L_sup_s = F.cross_entropy(logits_s, y_s)
        L_sup_t = F.cross_entropy(logits_t, y_t)
        L_sup_inner = L_sup_s + self.multiplier * L_sup_t

        self._update_norm_proto_ema(z_s, y_s, domain="s")
        self._update_norm_proto_ema(z_t, y_t, domain="t")

        w_s, w_t = self._calcul_weight(model, z_s, y_s, z_t, y_t, logits_s, logits_t)

        # discriminator step
        if isinstance(self.align, DANNLoss) and (self.opt_disc is not None):
            total = z_s.new_tensor(0.0)
            denom = z_s.new_tensor(0.0)

            for c in range(self.K):
                ms = (y_s == c)
                mt = (y_t == c)
                if (not ms.any()) or (not mt.any()):
                    continue

                z_s_c = z_s.detach()[ms]
                z_t_c = z_t.detach()[mt]
                w_s_c = w_s[ms]
                w_t_c = w_t[mt]

                Bs_c = z_s_c.shape[0]
                Bt_c = z_t_c.shape[0]

                z_dom = torch.cat([z_s_c, z_t_c], dim=0)
                Bc, Vv, Dd = z_dom.shape
                zf = z_dom.reshape(Bc * Vv, Dd)

                d = torch.cat([
                    torch.zeros(Bs_c, device=device, dtype=torch.long),
                    torch.ones(Bt_c, device=device, dtype=torch.long),
                ], dim=0).unsqueeze(1).expand(Bc, Vv).reshape(Bc * Vv)

                w_dom = torch.cat([w_s_c, w_t_c], dim=0).reshape(Bc * Vv).clamp_min(1e-12)
                w_dom = w_dom / w_dom.sum()

                logits_d = self.align.disc(zf)
                ce = F.cross_entropy(logits_d, d, reduction="none")
                loss_c = (w_dom * ce).sum()

                alpha = z_s.new_tensor(float(Bs_c + Bt_c))
                total = total + alpha * loss_c
                denom = denom + alpha

            disc_loss = total / denom.clamp_min(1e-12)
            self.opt_disc.zero_grad(set_to_none=True)
            disc_loss.backward()
            self.opt_disc.step()

        L_align = self._align(z_s, z_t, w_s, w_t, y_s=y_s, y_t=y_t)
        L_inner = L_sup_inner + self.lambda_align * L_align

        optimizer.zero_grad(set_to_none=True)
        L_inner.backward()
        optimizer.step()

        return {
            "c": float((L_sup_s.item() + L_sup_t.item()) * 0.5),
            "a": float(L_align.item()),
        }


def build_method(args, method: str, feat_dim: int, num_classes: int):
    return MAA(args, feat_dim, num_classes)