
# %%
import torch
import torch.nn as nn
import numpy as np

# %%
class SurvivalModule(nn.Module):

    def __init__(self, monotonic_net, t_scaling):
        super().__init__()
        self.Λ = monotonic_net
        self.t_scaling = t_scaling


    def scale_ts(self, ts):
        ts = ts / self.t_scaling

        assert torch.all(ts >= 0), "Some ts are negative."
        assert ts.shape == (ts.shape[0], 1), f"{ts.shape=} != {(ts.shape[0], 1)=}"
        return ts


    def _check_S_t_and_clamp(self, S_t, ts):
        assert S_t.shape == (ts.shape[0], 1), f"{S_t.shape=} != {ts.shape=}"
        assert torch.all(-1e-2 < S_t), f"{torch.min(S_t)=}"
        assert torch.all(S_t < 1 + 1e-2), f"{torch.max(S_t)=}"
        return torch.clamp(S_t, 0, 1)


    def _check_λ_and_clamp(self, λ):
        assert torch.all(-1e-2 < λ), f"{torch.min(λ)=}"
        λ =  torch.clamp(λ, 0, np.inf)
        return λ


    def forward(self, ts, xs=None):
        ts = self.scale_ts(ts)

        λ = self.Λ(t=ts, z=xs)
        λ = self._check_λ_and_clamp(λ)

        S_t = torch.exp(-λ)
        S_t = self._check_S_t_and_clamp(S_t, ts)
        return S_t

# %%



