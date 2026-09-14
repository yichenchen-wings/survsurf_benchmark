"""Base Lightning module shared by the benchmark model wrappers."""

import torch
import lightning.pytorch as pl

STR_VAL_LOSS = 'val_loss'

class LitModel(pl.LightningModule):
    """Provide common optimization, loss logging, and epoch aggregation.

    Subclasses implement ``forward`` and may override validation when they consume
    the paired observed/probability-grid loaders returned by data modules.
    """
    def __init__(self, model, loss_fn, lr=0.001, print_epoch=False):
        """Initialize LitModel and store its configuration."""
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.epochs_run = 0 # for grad accum across batches
        self.print_epoch = print_epoch

    def forward(self,batch):
        # xs, slice_loc, ys, weights = batch
        # return self.model(xs)[0]
        """Run model inference for the supplied batch or tensor."""
        raise NotImplementedError

    def training_step(self, batch, batch_idx): #batch_idx is a compulsory argument
        # Compute the loss
        """Compute and retain the loss for one training batch."""
        loss = self.loss_fn(self, batch)

        eval_res = dict()
        eval_res['loss'] = loss.detach()
        eval_res['batch_size'] = batch[-1].shape[0]
        self.training_step_outputs.append(eval_res)
        return loss

    def configure_optimizers(self):
        """Create the Adam optimizer used by Lightning."""
        params=self.model.parameters()
        optimizer = torch.optim.Adam(
            params,
            lr=self.lr
        )
        return optimizer

    def test_step(self, batch, batch_idx): #batch_idx is a compulsory argument
        # Compute the loss
        """Compute and log loss for one test batch."""
        loss = self.loss_fn(self, batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return {'loss':loss.detach()}


    def validation_step(self, batch, batch_idx): #batch_idx is a compulsory argument:

        # Compute the loss
        """Evaluate one validation batch and retain its outputs."""
        loss = self.loss_fn(self, batch)

        eval_res = dict()
        eval_res['loss'] = loss.detach()
        eval_res['batch_size'] = batch[-1].shape[0]
        self.validation_step_outputs.append(eval_res)

    def on_train_epoch_end(self):
        """Aggregate and log the sample-weighted training loss."""
        self.epochs_run += 1
        N = sum(output['batch_size'] for output in self.training_step_outputs)
        train_loss = sum(output['loss']*output['batch_size'] for output in self.training_step_outputs) / N
        self.log("train_loss", train_loss)

        if self.print_epoch:
            message = f'EPOCH:{self.epochs_run} training loss: {train_loss.item()} '
            print(message)

        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        """Aggregate and log results from all validation loaders."""
        N = sum(output['batch_size'] for output in self.validation_step_outputs)
        val_loss = sum(output['loss']*output['batch_size'] for output in self.validation_step_outputs) / N
        self.log(STR_VAL_LOSS, val_loss)

        if self.print_epoch:
            message = f'EPOCH:{self.epochs_run} val loss: {val_loss.item()}'
            print(message)

        self.validation_step_outputs.clear()
