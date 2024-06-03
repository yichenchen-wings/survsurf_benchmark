import torch
from soren_survcurv import CoxTimeDependentNN
from soren_survcurv.losses import SuMoLoss
from soren_survcurv.nets.monotone_module import MonotonicIncreasingNet, MonotonicIncreasingVectorNet
import numpy as np


def get_CoxTimeDependentNN(
        n_input_feats_g_excl,
        t_max,
        n_hidden_layers=5,
        n_hidden_dim=32,
):

    monotonic_increasing_net = MonotonicIncreasingNet(latent_sizes=[n_hidden_dim]*n_hidden_layers)
    monotonic_increasing_net_coefficients = MonotonicIncreasingVectorNet(latent_sizes=[n_hidden_dim]*n_hidden_layers + [n_input_feats_g_excl+1,])

    model = CoxTimeDependentNN(
        n_input_features=n_input_feats_g_excl+1,
        monotonic_increasing_net_baseline=monotonic_increasing_net,
        monotonic_increasing_net_coefficients=monotonic_increasing_net_coefficients,
        t_scaling=t_max, #horizon
    )
    return model



def loss_brier(model, batch):
    subjects, Xs, ts, ys = batch
    outputs = model(batch)
    loss_fn = torch.nn.functional.mse_loss
    ys = ys.type(outputs.dtype)

    losses = loss_fn(1-outputs, ys)
    return losses

def loss_sumo(model, batch):
    subjects, Xs, ts, ys = batch
    outputs = model(batch)
    loss_fn = SuMoLoss()
    ys = ys.type(outputs.dtype)

    losses = loss_fn(outputs, 1-ys, ts)
    return losses

def loss_bce(model, batch):
    subjects, Xs, ts, ys = batch
    outputs = model(batch)
    loss_fn = torch.nn.functional.binary_cross_entropy
    ys = ys.type(outputs.dtype)

    losses = loss_fn(1-outputs, ys, ts)
    return losses

def loss_dydt(model, batch): # same as sumo
    subjects, Xs, ts, ys = batch
    ts.requires_grad_()
    outputs = model(batch)

    dydt = torch.autograd.grad(
        outputs=outputs,
        inputs=ts,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]
    dydt = torch.clamp(dydt, -np.inf, -1e-6)
    outputs = torch.clamp(outputs, 1e-6, 1-1e-6)
    losses = (
        ys*torch.log(-dydt) # if observed g (g > 0) at t, then dy/dt (i.e. prob of g occurring by or at t) for (t,g,x) should be high.
        + (1-ys)*torch.log(outputs) 
    )
    losses = -torch.mean(losses)
    return losses


def loss_dydg(model, batch):
    small_change = 1e-6

    subjects, Xs, ts, ys = batch
    outputs = 1-model(batch)

    Xs_g_greater = Xs.clone()
    Xs_g_greater[:,-1] = Xs_g_greater[:,-1] + small_change
    batch_g_greater = (subjects, Xs_g_greater, ts, ys)
    
    outputs_g_greater = 1-model(batch_g_greater)

    dydg = (outputs_g_greater - outputs)/small_change
    dydg = torch.clamp(dydg, -np.inf, -small_change)
    outputs = torch.clamp(outputs, small_change, 1-small_change)
    losses = (
        ys*torch.log(outputs)
        + ys*torch.log(-dydg) # if observed g (g > 0) at t, then dy/dg (i.e. prob of g occurring by or at t) for (t,g,x) should be high.
        + (1-ys)*torch.log(1-outputs) # if g = 0 at t, prob at g=0 cannot be computed, but (t, g_min/2, x), g_min > 0, should have prob closer to 0.
    )
    losses = -torch.mean(losses)
    return losses


from pl_wrapper import STR_VAL_LOSS, LitModel

class LitModelCoxTimeDependentNN(LitModel):
    def __init__(self, model, loss_fn, lr=0.001, print_epoch=False):
        super().__init__(
            model=model, 
            loss_fn=loss_fn, 
            lr=lr, 
            print_epoch=print_epoch
        )
        self.validation_loss = []
        self.validation_brier_on_probs = []
    def forward(self,batch):
        subjects, Xs, ts, ys = batch
        return self.model(Xs, ts)
    
    def validation_step(self, batch, batch_idx, dataloader_idx): #batch_idx is a compulsory argument:
        with torch.set_grad_enabled(True):
            if dataloader_idx == 0:
                # Compute the loss
                loss = self.loss_fn(self, batch)
                eval_res = dict()
                eval_res['loss'] = loss.detach()
                self.validation_loss.append(eval_res)
            if dataloader_idx == 1:
                # Compute the loss
                loss = loss_brier(self, batch)
                eval_res = dict()
                eval_res['brier_on_probs'] = loss.detach()
                self.validation_brier_on_probs.append(eval_res)

    def on_validation_epoch_end(self):
        val_loss = sum(output['loss'] for output in self.validation_loss) / len(self.validation_loss)
        self.log(STR_VAL_LOSS, val_loss)

        if self.print_epoch:
            message = f'EPOCH:{self.epochs_run} val loss: {val_loss.item()}'
            print(message)

        
        val_brier = sum(output['brier_on_probs'] for output in self.validation_brier_on_probs) / len(self.validation_brier_on_probs)
        self.log('val_score_brier', val_brier)

        if self.print_epoch:
            message = f'EPOCH:{self.epochs_run} val brier: {val_loss.item()}'
            print(message)

        self.validation_loss.clear()
        self.validation_brier_on_probs.clear()
