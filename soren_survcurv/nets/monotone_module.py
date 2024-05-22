# %% [markdown]
# # Monotonic increasing NNs

# %%
import torch
import numpy as np
import torch.nn as nn

# %% [markdown]
# # MonotonicIncreasingNet

# %%
class SoftplusSimpleMonotonic(nn.Module):

    def __init__(self, ω=1, β=0, output_size=64):
        super().__init__()
        self.output_size = output_size
        self.w_ = nn.Parameter(torch.rand(1, output_size) * ω)
        self.b  = nn.Parameter(torch.rand(1, output_size) * β)


    @property
    def w(self):
        self.w_.data.clamp_(0)
        return self.w_


    def transform(self, t):
        assert t.shape == (len(t), 1), f"{t.shape=}"
        return nn.functional.softplus(self.w * t - self.b)


    def forward(self, t):
        y = self.transform(t)
        assert t.shape == (t.shape[0], 1), f"{t.shape=}"
        assert y.shape == (y.shape[0], self.output_size), f"{y.shape=}"
        return y

# %%
class MonotonicIncreasingLayer(nn.Module):

    def __init__(self, input_size, z0_input_size, output_size, act, resblock=True):
        super().__init__()
        self.input_size = input_size
        self.z0_input_size = z0_input_size
        self.output_size = output_size
        self.resblock = resblock

        self.act = act
        self.simple_monotone_fct = SoftplusSimpleMonotonic()

        self.B = self._get_B(input_size, output_size)
        self.G = self._get_G(input_size, self.simple_monotone_fct.output_size)
        self.A = self._get_A(self.simple_monotone_fct.output_size, output_size)
        self.L = nn.Linear(z0_input_size, output_size)

        if resblock:
            self.H = self._get_H(input_size, output_size)


    def _get_B(self, input_size, output_size, scale_init_weight=0.2, scale_init_bias_B=-8.5):
        B = nn.Linear(input_size, output_size)
        B.weight.data = B.weight.data.abs() * scale_init_weight
        B.bias.data = torch.rand_like(B.bias) * scale_init_bias_B
        return B


    def _get_G(self, input_size, output_size, scale_init_weight=0.2, scale_init_bias_G=10):
        G = nn.Linear(input_size, output_size, bias=True)
        G.weight.data = G.weight.data.abs() * scale_init_weight
        G.bias.data = torch.rand_like(G.bias) * scale_init_bias_G
        return G


    def _get_A(self, input_size, output_size, scale_init_weight=0.2):
        A = nn.Linear(input_size, output_size, bias=False)
        A.weight.data = A.weight.data.abs() * scale_init_weight
        return A


    def _get_H(self, input_size, output_size, scale_init_weight=0.2):
        H = nn.Linear(input_size, output_size)
        H.weight.data = H.weight.data.abs() * scale_init_weight
        return H


    @torch.no_grad()
    def _clamp_weights(self):
        self.B.weight.data.clamp_(0)
        self.A.weight.data.clamp_(0)
        self.G.weight.data.clamp_(0)

        if self.resblock:
            self.H.weight.data.clamp_(0)


    def forward(self, z, z0, t):
        assert z.shape == (z.shape[0], self.input_size), f"{z.shape=}, {z.shape=}, {(z.shape[0], self.input_size)=}"
        assert t.shape == (z.shape[0], 1)
        assert torch.all(t >= 0)

        self._clamp_weights()

        γ = self.simple_monotone_fct(t)
        Gz = self.G(z)
        γGz = γ * nn.functional.softplus(Gz)
        AγGz = self.A(γGz)

        Bz = self.B(z)
        Lz0 = self.L(z0)
        z_new = self.act(Bz + AγGz + Lz0)

        if self.resblock:
            z_new = self.H(z) + z_new

        assert z_new.shape == (z.shape[0], self.output_size)
        return z_new

# %%
class MonotonicIncreasingVectorNet(nn.Module):

    def __init__(self, latent_sizes):
        super().__init__()
        self.sizes = latent_sizes

        self.layers = nn.ModuleList([])

        for i in range(len(self.sizes) - 1):
            is_last = (i == len(self.sizes) - 2)
            act = nn.Identity() if is_last else nn.Tanh()

            layer = MonotonicIncreasingLayer(
                input_size=self.sizes[i],
                z0_input_size=self.sizes[0],
                output_size=self.sizes[i+1],
                act=act,
            )

            self.layers.append(layer)


    def _adjust_towards_zero(self, z, z0, t):
        z = z - self(t=torch.zeros_like(t), z=z0, survival=False)

        if np.isnan(torch.min(z).item()):
            raise ValueError("Found a nan in one of MonotonicIncreasingVectorNet's activations.")
        assert torch.all(-1e-2 < z), f"{torch.min(z)=}"

        z = torch.clamp(z, 0, np.inf)
        return z


    def forward(self, t, z=None, survival=True):
        assert t.shape == (t.shape[0], 1)
        assert torch.all(t >= 0)

        if z is None:
            z = torch.zeros(t.shape[0], self.sizes[0], device=t.device)

        z0 = z.clone()

        for layer in self.layers:
            z = layer(z=z, z0=z0, t=t)

        if survival:
            z = self._adjust_towards_zero(z, z0, t)

        assert z.shape == (t.shape[0], self.sizes[-1])
        return z

# %%
class MonotonicIncreasingNet(MonotonicIncreasingVectorNet):

    def __init__(self, latent_sizes=[32]*5):
        super().__init__(latent_sizes=latent_sizes + [1])

# %%
class Exp(nn.Module):

    def forward(self, x):
        return torch.exp(x)

# %%



