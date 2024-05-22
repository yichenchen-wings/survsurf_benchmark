# %%
import torch.nn as nn

# %%
from .survival_module import SurvivalModule
from .monotone_module import MonotonicIncreasingNet

# %%
class SurvivalNet(nn.Module):

    def __init__(self, name, feature_model, survival_module):
        super().__init__()
        self.survival_module = survival_module
        self.feature_model = feature_model
        self.name = name


    def forward(self, ts, xs):
        ts.requires_grad_(True)
        zs = self.feature_model(xs)
        S_t = self.survival_module(ts=ts, xs=zs)
        return S_t

# %%
class SurvivalNetAssembler(SurvivalNet):

    def __init__(self,  n_input_features, n_latent_features, t_scaling, depth, width, depth_fm, width_fm):
        name = self._get_name()
        feature_model = self._get_feature_model(n_input_features, n_latent_features, depth_fm, width_fm)
        survival_module = self._get_survival_module(n_latent_features, depth, width, t_scaling)

        super().__init__(name, feature_model, survival_module)


    def _get_name(self):
        raise NotImplemented()


    def _get_feature_model(self, n_input_features, n_latent_features, depth_fm, width_fm):
        raise NotImplemented()


    def _get_survival_module(self, n_latent_features, depth, width, t_scaling):
        raise NotImplemented()

# %%
class SurvivalNN(SurvivalNetAssembler):

    def __init__(self, n_input_features, n_latent_features, t_scaling, depth=5, width=32, depth_fm=3, width_fm=32):
        super().__init__(n_input_features, n_latent_features, t_scaling, depth, width, depth_fm, width_fm)


    def _get_name(self):
        return "SurvivalNN"


    def _get_feature_model(self, n_input_features, n_latent_features, depth_fm, width_fm):
        depth=depth_fm
        width=width_fm

        layers = []
        layers.append(nn.Linear(n_input_features, width))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(width))

        for i in range(depth-2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(width))

        layers.append(nn.Linear(width, n_latent_features))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(n_latent_features))

        return nn.Sequential(*layers)


    def _get_survival_module(self, n_latent_features, depth, width, t_scaling):
        monotonic_net = MonotonicIncreasingNet(latent_sizes=[n_latent_features] + [width]*depth)
        return SurvivalModule(monotonic_net, t_scaling)

# %%



