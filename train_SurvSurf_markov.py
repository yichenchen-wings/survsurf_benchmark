# %%
import argparse

import time
start_time = time.time()


parser = argparse.ArgumentParser(description='Train a SurvSurf model on a dataset')
parser.add_argument(
    '-s', '--seed', 
    type=int,
    default=10,
    help='random seed to use'
)
args = parser.parse_args()


# %%
import os

# %% [markdown]
# # Import and instantiate model

# %%
import wandb

from pl_train import pl_wdb_train
wandb.login()



PROJ_NAME = 'SurvSurfBenchmark_Markov'
SEEDS = [args.seed]

config = {
    'dropout':None,
    'dir_runtime_results':'./runtime_results',
    'model_getter':'get_SurvSurf',
    'ds_name':'markov_32feat_11t5g_more_balanced',
    'datamodule':'DataModuleMarkovSurvSurf',
    'train_mode':'full_traj_obs_only',
    'eval_mode':'true_probs_grid',
    'loss':'LossSumo',
    'n_hidden_layers':64,
    'n_hidden_dim':8,
    'g_resol':0.5,
    'batch_size':64,
    'patience':50,
    'max_epoch':200,
    'accum_grad_batches':1,
    'lr':1e-3,
    'weight_decay':2e-2,
    'device':'cpu',
    'save_top_k':1,
    't_res_in_loss':1, 
    't_res_at_trans':1,
    'watch_model':False, # if True then model checkpoint will not be compatible to the pytorch-lightning model wrapper TODO: investigate when have time
}

from pl_wrapper import STR_VAL_LOSS
from model_factory_survsurf import LitModelSurvSurf
import model_factory_survsurf
model_getter = model_factory_survsurf.__dict__[config['model_getter']]

import torch
torch.set_float32_matmul_precision('medium')


# %%

# %% [markdown]
# # Get ready for training
for seed in SEEDS:
    from monotonic_nn_surv_surf import __version__
    config['seed'] = seed
    config['survsurf_ver'] = __version__

    run = wandb.init(
        project=PROJ_NAME,
        save_code=True,
        config=config,
        name='__SEED'.join([config['model_getter'], str(seed)])
    )
    run.log_code(".")
    
    wandb.define_metric(STR_VAL_LOSS, summary="min")

    import lightning.pytorch as pl
    pl.seed_everything(seed=run.config.seed)
    
    import datasets
    data_module_cls = datasets.__dict__[config['datamodule']]

    datamodule = data_module_cls(
        df_dir='/home/yc366/repos/survsurf_benchmark/dataset_split', 
        ds_name=run.config.ds_name, 
        g_resol=run.config.g_resol, 
        separate_g_from_feats=True, 
        batch_size=run.config.batch_size, 
        num_workers=4,
        t_resol=run.config.t_res_at_trans,
        train_mode=run.config.train_mode,
        eval_mode=run.config.eval_mode
    )


    model = model_getter(
        n_input_feats_g_excl=datamodule.n_feats_g_excl,
        t_max=10,
        n_hidden_layers=run.config.n_hidden_layers,
        n_hidden_dim=run.config.n_hidden_dim,
    )

    loss_fn_cls = model_factory_survsurf.__dict__[config['loss']]
    loss_fn = loss_fn_cls(t_res=run.config.t_res_in_loss)

    model_lit = LitModelSurvSurf(
        model=model, 
        loss_fn=loss_fn, 
        lr=run.config.lr,
        weight_decay=run.config.weight_decay
    )

    if run.config.watch_model:
        module_to_log = model_lit
        print(f'watching model...')
    else:
        module_to_log = None

    pl_wdb_train(
        dir_runtime_results=run.config.dir_runtime_results, 
        patience=run.config.patience, 
        max_epoch=run.config.max_epoch, 
        accumulate_grad_batches=run.config.accum_grad_batches,
        device=run.config.device, 
        proj_name=PROJ_NAME, 
        save_top_k=run.config.save_top_k, 
        datamodule=datamodule, 
        model_lit=model_lit,
        module_to_log=module_to_log,
        log_every_n_batches=run.config.accum_grad_batches,
        inference_mode=True
    )
    run.finish()



time_taken_s = int(time.time() - start_time)
hr = time_taken_s//(3600)
hr_in_s = hr*3600
minutes = (time_taken_s - hr_in_s)//60
min_in_s = minutes*60
seconds = time_taken_s - hr_in_s - min_in_s

print('='*10 + 'END' + '='*10)
script_str = 'training'
print(f'Time taken to run {script_str} script: {str(hr).zfill(2)}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}')



