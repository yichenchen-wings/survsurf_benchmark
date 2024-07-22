import torch
from torch import nn
from torch_deephit.deephit import DeepHit
import numpy as np


def get_DeepHit(
        n_input_feats_g_excl,
        t_size,
        h_dim_shared=None,
        h_dim_cause_spcfc=None,
        n_layers_shared=None,
        n_layers_cause_spcfc=None,
        dropout=None

):
    n_input_feats = n_input_feats_g_excl+1
    if h_dim_shared is None:
        h_dim_shared = n_input_feats*3
    if h_dim_cause_spcfc is None:
        h_dim_cause_spcfc = n_input_feats*5
    if n_layers_shared is None:
        n_layers_shared = 1
    if n_layers_cause_spcfc is None:
        n_layers_cause_spcfc = 2
    if dropout is None:
        n_layers_shared = 0
    model = DeepHit(
        n_input_feats=n_input_feats, 
        k_compete_events=1, 
        t_size=t_size, 
        h_dim_shared=h_dim_shared, 
        h_dim_cause_spcfc=h_dim_cause_spcfc, 
        n_layers_shared=n_layers_shared, 
        n_layers_cause_spcfc=n_layers_cause_spcfc, 
        act_cls=nn.ReLU, 
        dropout=dropout
    )
    return model

class LossBrierDeepHitTrans:
    def __init__(self, t_size, t_res, g_res=None):
        self.t_size = t_size
        self.t_res = t_res
        self.tmax_allowed = self.t_size*self.t_res

    def loss_brier(self, model, batch):
        subjects, Xs, ts, ys, weight = batch

        torch._assert(ts.dim() == 2, 'ts should have two dimensions (bs, 1)')
        torch._assert(ts.shape[-1] == 1, 'ts should have two dimensions (bs, 1)')
        
        torch._assert(ys.dim() == 2, 'ys should have two dimensions (bs, 1)')
        torch._assert(ys.shape[-1] == 1, 'ys should have two dimensions (bs, 1)')

        torch._assert(torch.all(ts <= self.tmax_allowed), f'found input time beyond allowed horizon {self.tmax_allowed}.')
        
        outputs = model(batch) 
        bs, k, t_size = outputs.shape
        torch._assert(k == 1, f'found competing events (output shape = {outputs.shape}), not supported by current loss.')
        torch._assert(t_size == self.t_size, f'found mismatching time(last) dim (output shape = {outputs.shape}).')

        
        ts = ts.repeat(1, self.t_size).reshape(outputs.shape)
        CIF = torch.cumsum(outputs, dim=-1)
        t_matrix = torch.linspace(0, self.t_size-1, steps=self.t_size)*self.t_res
        t_matrix = t_matrix.repeat(Xs.shape[0], 1)
        t_matrix = t_matrix.reshape(*outputs.shape)
        truth_matrix = t_matrix.clone()
        truth_matrix[t_matrix < ts] = 0
        truth_matrix[t_matrix >= ts] = 1

        selector_not_censored = ys == 1
        selector_not_censored = selector_not_censored.repeat(1, t_matrix.shape[-1])
        selector_not_censored = selector_not_censored.reshape(*outputs.shape)


        selector_censored = ys == 0
        selector_censored = selector_censored.repeat(1, t_matrix.shape[-1])
        selector_censored = selector_censored.reshape(*outputs.shape)
        selector_censored = torch.logical_and(selector_censored, (t_matrix <= ts))

        loss = torch.sum(
            selector_not_censored*torch.square(truth_matrix - CIF) + selector_censored*torch.square(CIF)
        )
        loss = loss/(torch.sum(selector_not_censored) + torch.sum(selector_censored))
        return loss
    
    def __call__(self, model, batch):
        return self.loss_brier(model, batch)
    

class LossDyDgEmphPos:
    def __init__(self, t_size, t_res, g_res):
        self.t_size = t_size
        self.t_res = t_res
        self.g_resol = g_res
        self.tmax_allowed = self.t_size*self.t_res

    def loss_dydg(self, model, batch):
        subjects, Xs, ts, ys, weight = batch

        torch._assert(ts.dim() == 2, 'ts should have two dimensions (bs, 1)')
        torch._assert(ts.shape[-1] == 1, 'ts should have two dimensions (bs, 1)')
        
        torch._assert(ys.dim() == 2, 'ys should have two dimensions (bs, 1)')
        torch._assert(ys.shape[-1] == 1, 'ys should have two dimensions (bs, 1)')

        torch._assert(torch.all(ts <= self.tmax_allowed), f'found input time beyond allowed horizon {self.tmax_allowed}.')

        Xs_greater_g = Xs.clone()
        Xs_greater_g[:, -1] = Xs_greater_g[:, -1] + self.g_resol
        batch_greater_g = (subjects, Xs_greater_g, ts, ys, weight)
        
        outputs = model(batch) 
        outputs_greater_g = model(batch_greater_g) 
        bs, k, t_size = outputs.shape
        torch._assert(k == 1, f'found competing events (output shape = {outputs.shape}), not supported by current loss.')
        torch._assert(t_size == self.t_size, f'found mismatching time(last) dim (output shape = {outputs.shape}).')

        outputs = torch.clamp(outputs, 1e-6, 1-1e-6)
        CIF = torch.cumsum(outputs, dim=-1)
        CIF_greater_g = torch.cumsum(outputs_greater_g, dim=-1)

        dy = CIF_greater_g - CIF
        dy = torch.clamp(dy, -(1-1e-6), -1e-6)
        CIF = torch.clamp(CIF, 1e-6, 1-1e-6)
        CIF_greater_g = torch.clamp(CIF_greater_g, 1e-6, 1-1e-6)
        ts = ts.repeat(1, self.t_size).reshape(outputs.shape)
        t_matrix = torch.linspace(0, self.t_size-1, steps=self.t_size)*self.t_res
        t_matrix = t_matrix.repeat(Xs.shape[0], 1)
        t_matrix = t_matrix.reshape(*outputs.shape)
        t_selector = torch.zeros(t_matrix.shape)
        t_selector[torch.logical_and(t_matrix >= ts, t_matrix < (ts + self.t_res))] = 1
        assert torch.all(t_selector.sum(dim=-1) == 1)

        truth_matrix = ys.repeat(1, self.t_size).reshape(outputs.shape)
        losses = torch.sum(
            truth_matrix*t_selector*torch.log(CIF)
            +  truth_matrix*t_selector*torch.log(-dy) # if observed g (g > 0) at t, then dy/dg (i.e. prob of g occurring by or at t) for (t,g,x) should be high.
            + (1-truth_matrix)*t_selector*torch.log(1-CIF) # if g = 0 at t, prob at g=0 cannot be computed, but (t, g_min/2, x), g_min > 0, should have prob closer to 0.
        )
        loss = -losses/bs
        return loss
    
    def __call__(self, model, batch):
        return self.loss_dydg(model, batch)

class LossDyDg:
    def __init__(self, t_size, t_res, g_res):
        self.t_size = t_size
        self.t_res = t_res
        self.g_resol = g_res
        self.tmax_allowed = self.t_size*self.t_res

    def loss_dydg(self, model, batch):
        subjects, Xs, ts, ys, weight = batch

        torch._assert(ts.dim() == 2, 'ts should have two dimensions (bs, 1)')
        torch._assert(ts.shape[-1] == 1, 'ts should have two dimensions (bs, 1)')
        
        torch._assert(ys.dim() == 2, 'ys should have two dimensions (bs, 1)')
        torch._assert(ys.shape[-1] == 1, 'ys should have two dimensions (bs, 1)')

        torch._assert(torch.all(ts <= self.tmax_allowed), f'found input time beyond allowed horizon {self.tmax_allowed}.')

        Xs_greater_g = Xs.clone()
        Xs_greater_g[:, -1] = Xs_greater_g[:, -1] + self.g_resol
        batch_greater_g = (subjects, Xs_greater_g, ts, ys, weight)
        
        outputs = model(batch) 
        outputs_greater_g = model(batch_greater_g) 
        bs, k, t_size = outputs.shape
        torch._assert(k == 1, f'found competing events (output shape = {outputs.shape}), not supported by current loss.')
        torch._assert(t_size == self.t_size, f'found mismatching time(last) dim (output shape = {outputs.shape}).')

        outputs = torch.clamp(outputs, 1e-6, 1-1e-6)
        CIF = torch.cumsum(outputs, dim=-1)
        CIF_greater_g = torch.cumsum(outputs_greater_g, dim=-1)

        dy = CIF_greater_g - CIF
        dy = torch.clamp(dy, -(1-1e-6), -1e-6)
        CIF = torch.clamp(CIF, 1e-6, 1-1e-6)
        CIF_greater_g = torch.clamp(CIF_greater_g, 1e-6, 1-1e-6)
        ts = ts.repeat(1, self.t_size).reshape(outputs.shape)
        t_matrix = torch.linspace(0, self.t_size-1, steps=self.t_size)*self.t_res
        t_matrix = t_matrix.repeat(Xs.shape[0], 1)
        t_matrix = t_matrix.reshape(*outputs.shape)
        t_selector = torch.zeros(t_matrix.shape)
        t_selector[torch.logical_and(t_matrix >= ts, t_matrix < (ts + self.t_res))] = 1
        assert torch.all(t_selector.sum(dim=-1) == 1)

        truth_matrix = ys.repeat(1, self.t_size).reshape(outputs.shape)
        losses = torch.sum(
            truth_matrix*t_selector*torch.log(-dy) # if observed g (g > 0) at t, then dy/dg (i.e. prob of g occurring by or at t) for (t,g,x) should be high.
            + (1-truth_matrix)*t_selector*torch.log(1-CIF) # if g = 0 at t, prob at g=0 cannot be computed, but (t, g_min/2, x), g_min > 0, should have prob closer to 0.
        )
        loss = -losses/bs
        return loss
    
    def __call__(self, model, batch):
        return self.loss_dydg(model, batch)


class LossBrierDeepHit:
    def __init__(self, t_size, t_res, g_res=None):
        self.t_size = t_size
        self.t_res = t_res
        self.tmax_allowed = self.t_size*self.t_res

    def loss_brier(self, model, batch):
        subjects, Xs, ts, ys, weight = batch

        torch._assert(ts.dim() == 2, 'ts should have two dimensions (bs, 1)')
        torch._assert(ts.shape[-1] == 1, 'ts should have two dimensions (bs, 1)')
        
        torch._assert(ys.dim() == 2, 'ys should have two dimensions (bs, 1)')
        torch._assert(ys.shape[-1] == 1, 'ys should have two dimensions (bs, 1)')

        torch._assert(torch.all(ts <= self.tmax_allowed), f'found input time beyond allowed horizon {self.tmax_allowed}.')
        
        outputs = model(batch) 
        bs, k, t_size = outputs.shape
        torch._assert(k == 1, f'found competing events (output shape = {outputs.shape}), not supported by current loss.')
        torch._assert(t_size == self.t_size, f'found mismatching time(last) dim (output shape = {outputs.shape}).')

        
        ts = ts.repeat(1, self.t_size).reshape(outputs.shape)
        CIF = torch.cumsum(outputs, dim=-1)
        t_matrix = torch.linspace(0, self.t_size-1, steps=self.t_size)*self.t_res
        t_matrix = t_matrix.repeat(Xs.shape[0], 1)
        t_matrix = t_matrix.reshape(*outputs.shape)
        trans_selector = torch.zeros(t_matrix.shape)
        trans_selector[torch.logical_and(t_matrix >= ts, t_matrix < (ts + self.t_res))] = 1
        assert torch.all(trans_selector.sum(dim=-1) == 1)

        truth_matrix = ys.repeat(1, self.t_size).reshape(outputs.shape)

        loss = torch.sum(
            trans_selector*torch.square(truth_matrix - CIF)
        )
        loss = loss/bs
        return loss
    
    def __call__(self, model, batch):
        return self.loss_brier(model, batch)

class LossSumo:
    def __init__(self, t_size, t_res, g_res=None):
        self.t_size = t_size
        self.t_res = t_res
        self.tmax_allowed = self.t_size*self.t_res

    def loss_sumo(self, model, batch):
        subjects, Xs, ts, ys, weight = batch

        torch._assert(ts.dim() == 2, 'ts should have two dimensions (bs, 1)')
        torch._assert(ts.shape[-1] == 1, 'ts should have two dimensions (bs, 1)')
        
        torch._assert(ys.dim() == 2, 'ys should have two dimensions (bs, 1)')
        torch._assert(ys.shape[-1] == 1, 'ys should have two dimensions (bs, 1)')

        torch._assert(torch.all(ts <= self.tmax_allowed), f'found input time beyond allowed horizon {self.tmax_allowed}.')
        
        outputs = model(batch) 
        bs, k, t_size = outputs.shape
        torch._assert(k == 1, f'found competing events (output shape = {outputs.shape}), not supported by current loss.')        
        torch._assert(t_size == self.t_size, f'found mismatching time(last) dim (output shape = {outputs.shape}).')

        ts = ts.repeat(1, self.t_size).reshape(outputs.shape)
        CIF = torch.cumsum(outputs, dim=-1)
        t_matrix = torch.linspace(0, self.t_size-1, steps=self.t_size)*self.t_res
        t_matrix = t_matrix.repeat(Xs.shape[0], 1)
        t_matrix = t_matrix.reshape(*outputs.shape)
        trans_selector = torch.zeros(t_matrix.shape)
        trans_selector[torch.logical_and(t_matrix >= ts, t_matrix < (ts + self.t_res))] = 1

        selector_not_censored = ys == 1
        selector_not_censored = selector_not_censored.repeat(1, t_matrix.shape[-1])
        selector_not_censored = selector_not_censored.reshape(*outputs.shape)

        selector_censored = ys == 0
        selector_censored = selector_censored.repeat(1, t_matrix.shape[-1])
        selector_censored = selector_censored.reshape(*outputs.shape)

        CIF = torch.clamp(CIF, 0, 1-1e-6)
        loss = torch.sum(
            selector_not_censored*trans_selector*torch.log(outputs) 
            + selector_censored*trans_selector*torch.log(1-CIF)
        )
        loss = -loss/bs
        return loss
    
    def __call__(self, model, batch):
        return self.loss_sumo(model, batch)
    



from pl_wrapper import STR_VAL_LOSS, LitModel

class LitModelDeepHit(LitModel):
    def __init__(self, model, loss_fn, t_size, t_res, weight_decay, lr=0.001, print_epoch=False):
        super().__init__(
            model=model, 
            loss_fn=loss_fn, 
            lr=lr, 
            print_epoch=print_epoch
        )
        self.validation_loss = []
        self.validation_brier_on_probs = []
        self.weight_decay = weight_decay
        self.loss_brier = LossBrierDeepHit(t_size, t_res)
    def forward(self,batch):
        subjects, Xs, ts, ys, weight = batch
        return self.model(Xs)
    
    def configure_optimizers(self):
        params=self.model.parameters()
        optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer
    
    def validation_step(self, batch, batch_idx, dataloader_idx): #batch_idx is a compulsory argument:
        if dataloader_idx == 0:
            # Compute the loss
            loss = self.loss_fn(self, batch)
            eval_res = dict()
            eval_res['loss'] = loss.detach()
            eval_res['batch_size'] = batch[0].shape[0]
            self.validation_loss.append(eval_res)
        if dataloader_idx == 1:
            # Compute the loss
            loss = self.loss_brier(self, batch)
            eval_res = dict()
            eval_res['brier_on_probs'] = loss.detach()
            eval_res['batch_size'] = batch[0].shape[0]
            self.validation_brier_on_probs.append(eval_res)

    def on_validation_epoch_end(self):
        N = sum(output['batch_size'] for output in self.validation_loss)
        val_loss = sum(output['loss']*output['batch_size'] for output in self.validation_loss) / N
        self.log(STR_VAL_LOSS, val_loss)

        if self.print_epoch:
            message = f'EPOCH:{self.epochs_run} val loss: {val_loss.item()}'
            print(message)

        N = sum(output['batch_size'] for output in self.validation_brier_on_probs)
        val_brier = sum(output['brier_on_probs']*output['batch_size'] for output in self.validation_brier_on_probs) / N        
        self.log('val_score_brier', val_brier)

        if self.print_epoch:
            message = f'EPOCH:{self.epochs_run} val brier: {val_loss.item()}'
            print(message)

        self.validation_loss.clear()
        self.validation_brier_on_probs.clear()
