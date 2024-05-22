import torch
from monotonic_nn_surv_surf.core.survival_surface_nn import SurvivalSurface
from monotonic_nn_surv_surf.core.monotonic_net import MonotonicNet
from soren_survcurv.losses import SuMoLoss

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



def loss_brier(model, batch):
    subjects, Xs, gs, ts, ys = batch
    outputs = model(batch)
    loss_fn = torch.nn.functional.mse_loss
    ys = ys.type(outputs.dtype)

    losses = loss_fn(outputs, ys)
    return losses

def loss_sumo(model, batch):
    subjects, Xs, gs, ts, ys = batch
    ts.requires_grad_()
    outputs = model(batch)
    loss_fn = SuMoLoss()
    ys = ys.type(outputs.dtype)

    losses = loss_fn(1-outputs, 1-ys, ts)
    return losses



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
    def forward(self,batch):
        subjects, Xs, gs, ts, ys = batch
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
