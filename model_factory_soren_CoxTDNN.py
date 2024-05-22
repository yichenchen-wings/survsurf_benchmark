import torch
from soren_survcurv import CoxTimeDependentNN
from soren_survcurv.losses import SuMoLoss
from soren_survcurv.nets.monotone_module import MonotonicIncreasingNet, MonotonicIncreasingVectorNet


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
