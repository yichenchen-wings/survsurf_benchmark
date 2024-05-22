# %% [markdown]
# # Cox Model


# %%
import copy
import torch
import torch.nn as nn

# %%
from .monotone_module import Exp

# %% [markdown]
# # CoxNN

# %%
class CoxNN(nn.Module):

    def __init__(self, n_input_features, monotonic_increasing_net, t_scaling):
        super().__init__()
        self.name = "CoxNN"
        self.n_input_features = n_input_features
        self.Λ_0 = monotonic_increasing_net
        self.t_scaling = t_scaling

        self.accelerator_model = self.get_accelerator_model()


    def get_accelerator_model(self):
        return nn.Sequential(
            torch.nn.Linear(self.n_input_features, 1, bias=False),
            Exp()
        )


    def forward(self, xs, ts):
        assert ts.shape == (ts.shape[0], 1)
        assert torch.all(ts >= 0), "found negative t"

        ts.requires_grad_(True)
        ts = ts / self.t_scaling

        Λ_0_ts = self.Λ_0(t=ts)
        acceleration = self.accelerator_model(xs)

        S_t = torch.exp(-acceleration * Λ_0_ts)

        assert S_t.shape == (ts.shape[0], 1)
        assert Λ_0_ts.shape == (ts.shape[0], 1), Λ_0_ts.shape
        assert acceleration.shape == (ts.shape[0], 1), acceleration.shape
        return S_t

# %% [markdown]
# # CoxTimeDependentNN

# %%
class CoxTimeDependentNN(nn.Module):

    def __init__(self, n_input_features, monotonic_increasing_net_baseline, monotonic_increasing_net_coefficients, t_scaling):
        super().__init__()
        self.name = "CoxTimeDependentNN"
        self.n_input_features = n_input_features
        self.Λ_0 = monotonic_increasing_net_baseline

        self.coeff_net_positive = copy.deepcopy(monotonic_increasing_net_coefficients)
        self.coeff_net_negative = copy.deepcopy(monotonic_increasing_net_coefficients)

        self.t_scaling = t_scaling
        self.offset = nn.Parameter(torch.randn(1, self.n_input_features) * .1)


    def accelerator_positive(self, ts, xs):
        coeff = self.coeff_net_positive(ts, survival=False)
        ys = torch.mean(coeff * torch.relu(xs + self.offset), dim=-1, keepdim=True)
        return torch.exp(ys)


    def accelerator_negative(self, ts, xs):
        coeff  = self.coeff_net_negative(ts, survival=False)
        coeff -= self.coeff_net_negative(torch.zeros_like(ts), survival=False)
        coeff += self.coeff_net_positive(torch.zeros_like(ts), survival=False)

        ys = torch.mean(coeff * torch.relu(-(xs + self.offset)), dim=-1, keepdim=True)
        return torch.exp(ys)


    def forward(self, xs, ts):
        assert torch.all(ts >= 0), "negative ts"
        assert ts.shape == (ts.shape[0], 1), f"{ts.shape=}"
        assert ts.shape[0] == xs.shape[0], f"{ts.shape=}, {xs.shape=}"

        ts.requires_grad_(True)
        ts = ts / self.t_scaling

        Λ_0_ts = self.Λ_0(t=ts)
        acceleration = self.accelerator_positive(ts, xs) * self.accelerator_negative(ts, xs)

        S_t = torch.exp(-acceleration * Λ_0_ts)

        assert acceleration.shape == (xs.shape[0], 1)
        assert Λ_0_ts.shape == (ts.shape[0], 1), Λ_0_ts.shape
        assert acceleration.shape == (ts.shape[0], 1), acceleration.shape
        assert S_t.shape == (ts.shape[0], 1), f"{S_t.shape=}, {ts.shape=}"
        return S_t

# %%



