import torch
from monotonic_nn_surv_surf.core.survival_surface_nn import SurvivalSurface
from monotonic_nn_surv_surf.core.monotonic_net import MonotonicNet
from soren_survcurv.losses import SuMoLoss
import numpy as np

class SurvSurfNormTG(SurvivalSurface):
    def __init__(self, net, t_max):
        super().__init__(net)
        self.t_max = t_max
    
    def forward(self, ts, gs, xs=None):
        return super().forward(ts/self.t_max, gs, xs)

def get_SurvSurf(
        n_input_feats_g_excl,
        n_hidden_layers,
        n_hidden_dim,
        t_max
):

    monotonic_net = MonotonicNet(latent_sizes=[n_input_feats_g_excl] + [n_hidden_dim]*n_hidden_layers + [1])

    model = SurvSurfNormTG(monotonic_net, t_max)
    return model


class LossBrierSimple:
    def __init__(self, t_res=None):
        pass
    def loss_brier(self, model, batch):
        subjects, Xs, gs, ts, ys, weight = batch
        outputs = model(batch)
        ys = ys.type(outputs.dtype)

        losses = torch.square(outputs - ys)*weight
        return torch.mean(losses)
    
    def __call__(self, model, batch):
        return self.loss_brier(model, batch)


class LossDyDt: # this is the Sumo loss
    def __init__(self, t_res):
        self.t_res = t_res
    def loss_dydt(self, model, batch): 
        subjects, Xs, gs, ts, ys, weight = batch
        ts.requires_grad_()
        outputs = model(batch)

        dydt = torch.autograd.grad(
            outputs=outputs,
            inputs=ts,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True
        )[0]
        dydt = torch.clamp(dydt, 1e-6, np.inf)
        outputs = torch.clamp(outputs, 1e-6, 1-1e-6)
        losses = 1*(
            ys*torch.log(dydt) # if observed g (g > 0) at t, then dy/dt (i.e. prob of g occurring by or at t) for (t,g,x) should be high.
            + (1-ys)*torch.log(1-outputs) 
        )
        losses = -torch.mean(losses)
        return losses
    
    def loss_dy_t_res(self, model, batch):
        subjects, Xs, gs, ts, ys, weight = batch
        outputs = model(batch)
        
        t_before =  ts - self.t_res
        t_before =  torch.clamp(t_before, 0, torch.inf)

        batch_t_before = (subjects, Xs, gs, t_before, ys, weight )
        outputs_t_before = model(batch_t_before)

        dy = outputs - outputs_t_before

        dy = torch.clamp(dy, 1e-6, 1-1e-6)
        outputs = torch.clamp(outputs, 1e-6, 1-1e-6)
        losses = 1*(
            ys*torch.log(dy) # if observed g (g > 0) at t, then dy/dt (i.e. prob of g occurring by or at t) for (t,g,x) should be high.
            + (1-ys)*torch.log(1-outputs) 
        )
        losses = -torch.mean(losses)
        return losses
    
    def __call__(self, model, batch):
        if self.t_res:
            return self.loss_dy_t_res(model, batch)
        else:
            return self.loss_dydt(model, batch)


class LossDyDtEmphPos:
    def __init__(self, t_res):
        self.t_res = t_res
    def loss_dydt(self, model, batch): 
        subjects, Xs, gs, ts, ys, weight = batch
        ts.requires_grad_()
        outputs = model(batch)

        dydt = torch.autograd.grad(
            outputs=outputs,
            inputs=ts,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True
        )[0]
        dydt = torch.clamp(dydt, 1e-6, np.inf)
        outputs = torch.clamp(outputs, 1e-6, 1-1e-6)
        losses = 1*(
            ys*torch.log(outputs)
            + ys*torch.log(dydt) # if observed g (g > 0) at t, then dy/dt (i.e. prob of g occurring by or at t) for (t,g,x) should be high.
            + (1-ys)*torch.log(1-outputs) 
        )
        losses = -torch.mean(losses)
        return losses
    
    def loss_dy_t_res(self, model, batch):
        subjects, Xs, gs, ts, ys, weight = batch
        outputs = model(batch)
        
        t_before =  ts - self.t_res
        t_before =  torch.clamp(t_before, 0, torch.inf)

        batch_t_before = (subjects, Xs, gs, t_before, ys, weight )
        outputs_t_before = model(batch_t_before)

        dy = outputs - outputs_t_before

        dy = torch.clamp(dy, 1e-6, 1-1e-6)
        outputs = torch.clamp(outputs, 1e-6, 1-1e-6)
        losses = 1*(
            ys*torch.log(outputs) 
            + ys*torch.log(dy) # if observed g (g > 0) at t, then dy/dt (i.e. prob of g occurring by or at t) for (t,g,x) should be high.
            + (1-ys)*torch.log(1-outputs) 
        )
        losses = -torch.mean(losses)
        return losses
    
    def __call__(self, model, batch):
        if self.t_res:
            return self.loss_dy_t_res(model, batch)
        else:
            return self.loss_dydt(model, batch)


class LossDyDgEmphPos:
    def __init__(self, t_res=None, g_res=1/5):
        self.g_res=g_res
    
    def loss_dy_g_res(self, model, batch):
        subjects, Xs, gs, ts, ys, weight = batch
        gs.requires_grad_()
        outputs = model(batch)
        greater_g = gs + self.g_res
        batch_greater_g = (subjects, Xs, greater_g, ts, ys, weight )
        outputs_greater_g  = model(batch_greater_g)
        dy = outputs_greater_g - outputs
        dy = torch.clamp(dy, -(1-1e-6), -1e-6)
        outputs = torch.clamp(outputs, 1e-6, 1-1e-6)
        losses = (
            ys*torch.log(outputs)
            + ys*torch.log(-dy) # if observed g (g > 0) at t, then dy/dg (i.e. prob of g occurring by or at t) for (t,g,x) should be high.
            + (1-ys)*torch.log(1-outputs) # if g = 0 at t, prob at g=0 cannot be computed, but (t, g_min/2, x), g_min > 0, should have prob closer to 0.
        )
        losses = -torch.mean(losses)
        return losses
    
    def __call__(self, model, batch):
        return self.loss_dy_g_res(model, batch)
    
class LossDyDg:
    def __init__(self, t_res=None, g_res=1/5):
        self.g_res=g_res
    
    def loss_dydg(self, model, batch):
        subjects, Xs, gs, ts, ys, weight = batch
        gs.requires_grad_()
        outputs = model(batch)

        dydg = torch.autograd.grad(
            outputs=outputs,
            inputs=gs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True
        )[0]
        dydg = torch.clamp(dydg, -np.inf, -1e-6)
        outputs = torch.clamp(outputs, 1e-6, 1-1e-6)
        losses = (
            ys*torch.log(-dydg) # if observed g (g > 0) at t, then dy/dg (i.e. prob of g occurring by or at t) for (t,g,x) should be high.
            + (1-ys)*torch.log(1-outputs) # if g = 0 at t, prob at g=0 cannot be computed, but (t, g_min/2, x), g_min > 0, should have prob closer to 0.
        )
        losses = -torch.mean(losses)
        return losses
    
    def loss_dy_g_res(self, model, batch):
        subjects, Xs, gs, ts, ys, weight = batch
        outputs = model(batch)
        greater_g = gs + self.g_res
        batch_greater_g = (subjects, Xs, greater_g, ts, ys, weight )
        outputs_greater_g  = model(batch_greater_g)
        dy = outputs_greater_g - outputs
        dy = torch.clamp(dy, -(1-1e-6), -1e-6)
        outputs = torch.clamp(outputs, 1e-6, 1-1e-6)
        losses = (
            ys*torch.log(-dy) # if observed g (g > 0) at t, then dy/dg (i.e. prob of g occurring by or at t) for (t,g,x) should be high.
            + (1-ys)*torch.log(1-outputs) # if g = 0 at t, prob at g=0 cannot be computed, but (t, g_min/2, x), g_min > 0, should have prob closer to 0.
        )
        losses = -torch.mean(losses)
        return losses
    
    def __call__(self, model, batch):
        if self.g_res:
            return self.loss_dy_g_res(model, batch)
        else:
            return self.loss_dydg(model, batch)


from pl_wrapper import STR_VAL_LOSS, LitModel

class LitModelSurvSurf(LitModel):
    def __init__(self, model, loss_fn, lr, weight_decay, print_epoch=False):
        super().__init__(
            model=model, 
            loss_fn=loss_fn, 
            lr=lr, 
            print_epoch=print_epoch
        )
        self.validation_loss = []
        self.validation_brier_on_probs = []
        self.weight_decay = weight_decay
        self.eval_fn = LossBrierSimple()
    def forward(self,batch):
        subjects, Xs, gs, ts, ys, weight = batch
        return self.model(ts, gs, Xs)
    
    def configure_optimizers(self):
        params=self.model.parameters()
        optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer
    
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
                loss = self.eval_fn(self, batch)
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
