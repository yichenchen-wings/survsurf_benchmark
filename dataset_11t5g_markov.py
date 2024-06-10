
from typing import Literal
from lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

ds_name_to_n_feats_mapping = {
    'markov_3feat11t5g_more_balanced':3,
    'markov_3feat11t5g_less_balanced':3,
    'markov_32feat_11t5g_more_balanced':32,
    'markov_32feat_11t5g_less_balanced':32
}

COLNAME_SURVIVAL_DURATION = "duration"
COLNAME_SURVIVAL_EVENT_OBSERVED ="event_observed"
COL_WEIGHT = 'weight'


def _get_censor_status_for_g(df_max_g_by_t_subj, g):
    g_max =  df_max_g_by_t_subj['g_max_by_time'].max()
    seletor_has_obs_greater_g = df_max_g_by_t_subj['g_max_by_time'] >= g
    seletor_max_g = df_max_g_by_t_subj['g_max_by_time'] == g_max
    t_last_obs = df_max_g_by_t_subj.loc[seletor_max_g,'t'].max()
    record = dict()
    if seletor_has_obs_greater_g.any():
        record['g'] = g
        record[COLNAME_SURVIVAL_DURATION] = t_last_obs
        record[COLNAME_SURVIVAL_EVENT_OBSERVED] = True # If there is a greater g at t, g must have occured before then
        return pd.Series(record)
    else:
        record['g'] = g
        record[COLNAME_SURVIVAL_DURATION] = t_last_obs
        record[COLNAME_SURVIVAL_EVENT_OBSERVED] = False # If there isn't a greater or equal g at t_max, then g has not occured.
        return pd.Series(record)



def get_pre_censoring_t_train(df_dir, ds_name, gs):
    path_max_g_by_t_obs = os.path.join(df_dir,f'{ds_name}__df_state_history_sampled_max_train.csv')
    df_max_g_by_t_obs = pd.read_csv(path_max_g_by_t_obs, index_col=0)
    df_out = []
    for g in gs:
        df = df_max_g_by_t_obs.groupby('subject').apply(lambda df: _get_censor_status_for_g(df, g))
        df_out.append(df)
    df_out = pd.concat(df_out)
    return df_out


class DatasetMarkovSurvSurf(Dataset):
    def __init__(
            self, 
            df_dir, 
            ds_name, 
            g_resol, 
            split: Literal['train', 'tune', 'val', 'test'], 
            mode:Literal['first_cross_obs_only', 'full_traj_obs_only', 'true_probs_grid'],
            separate_g_from_feats: bool,
            t_resol=1
        ):
        assert t_resol == 1
        self.split = split
        self.mode = mode
        self.path_df_feature_per_sub = os.path.join(df_dir,f'{ds_name}__df_features_{split}.csv')
        self.path_max_g_by_t_obs = os.path.join(df_dir,f'{ds_name}__df_state_history_sampled_max_{split}.csv')
        self.path_true_prob = os.path.join(df_dir,f'{ds_name}__df_true_prob_long_{split}.csv')
        self.g_resol = g_resol
        self.separate_g_from_feats = separate_g_from_feats
        self.colname_traj_id = 'subject'
        self.colname_time = 't'
        self.colname_g = 'g_max_by_time'

        self.subjects, self.X, self.g, self.t, self.y, self.weight= self._get_df_Xy()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        if self.separate_g_from_feats:
            return self.subjects[index], self.X[index], self.g[index], self.t[index], self.y[index], self.weight[index]
        else:
            return self.subjects[index], self.X[index], self.t[index], self.y[index], self.weight[index]

    def _single_traj_to_trans_time(self, df_single_traj):
        traj = pd.Series(
            df_single_traj[self.colname_g].values,
            index=df_single_traj[self.colname_time].values
        )
        if 0 in traj.index:
            assert traj[0] == 0
        else:
            traj[0] = 0
        
        traj = traj.sort_index()
        traj_trans = traj.diff()
        assert all(traj_trans.index.isin(traj.index))
        max_trans_to = traj.max()
        rows_event_df = []
        for time, trans_mag in traj_trans.items():
            if trans_mag > 0: # record when a state/grade is first reached
                trans_to = traj[time]
                rows_event_df.append(
                    {
                        COLNAME_SURVIVAL_EVENT_OBSERVED:True,
                        COLNAME_SURVIVAL_DURATION:time,
                        self.colname_g:trans_to
                    }
                )
        # at the latest obs, the more severe states 'have not yet been observed'
        rows_event_df.append(
            {
                COLNAME_SURVIVAL_EVENT_OBSERVED:False,
                COLNAME_SURVIVAL_DURATION:time,
                self.colname_g: max_trans_to + self.g_resol 
            }
        )
        return pd.DataFrame(rows_event_df)
    
    def _get_df_Xy_trans_obs(self):
        xs = pd.read_csv(self.path_df_feature_per_sub, index_col=0)
        assert self.colname_traj_id in xs.columns
        assert xs[self.colname_traj_id].nunique() == xs[self.colname_traj_id].size

        max_g_by_t_obs = pd.read_csv(self.path_max_g_by_t_obs, index_col=0)
        assert self.colname_traj_id in max_g_by_t_obs.columns
        
        df_trans_time = max_g_by_t_obs.groupby(self.colname_traj_id).apply(
            self._single_traj_to_trans_time
        ).reset_index(level=0)
        df_xy = df_trans_time.merge(xs, on=self.colname_traj_id, how='left')
        df_xy = df_xy.loc[df_xy[self.colname_g] > 0,:]
        df_xy = df_xy.reset_index(drop=True)
        df_xy[COL_WEIGHT] = 1
        return df_xy    

    def _single_traj_full_to_label(self, df_single_traj):
        rows_event_df = pd.DataFrame()
        rows_event_df[COLNAME_SURVIVAL_DURATION] = df_single_traj[self.colname_time].values
        rows_event_df[COLNAME_SURVIVAL_EVENT_OBSERVED] = True
        rows_event_df[self.colname_g] = df_single_traj[self.colname_g].values

        rows_event_df.loc[
            rows_event_df[self.colname_g] == 0,
            COLNAME_SURVIVAL_EVENT_OBSERVED
        ]  = False
        
        rows_event_df.loc[
            rows_event_df[self.colname_g] == 0,
            self.colname_g
        ]  = self.g_resol
        
        return rows_event_df
    
    def _get_df_Xy_full_traj_obs(self):
        xs = pd.read_csv(self.path_df_feature_per_sub, index_col=0)
        assert self.colname_traj_id in xs.columns
        assert xs[self.colname_traj_id].nunique() == xs[self.colname_traj_id].size

        max_g_by_t_obs = pd.read_csv(self.path_max_g_by_t_obs, index_col=0)
        assert self.colname_traj_id in max_g_by_t_obs.columns
        
        df_trans_time = max_g_by_t_obs.groupby(self.colname_traj_id).apply(
            self._single_traj_full_to_label
        ).reset_index(level=0)
        df_xy = df_trans_time.merge(xs, on=self.colname_traj_id, how='left')
        df_xy = df_xy.loc[df_xy[self.colname_g] > 0,:]
        df_xy = df_xy.reset_index(drop=True)
        df_xy[COL_WEIGHT] = 1
        return df_xy
    
    def _get_df_Xy_true_prob(self):
        xs = pd.read_csv(self.path_df_feature_per_sub, index_col=0)
        assert self.colname_traj_id in xs.columns
        assert xs[self.colname_traj_id].nunique() == xs[self.colname_traj_id].size

        true_probs_all_tg = pd.read_csv(self.path_true_prob, index_col=0)
        assert self.colname_traj_id in true_probs_all_tg.columns
        
        df_xy = true_probs_all_tg.merge(xs, on=self.colname_traj_id, how='left')
        df_xy = df_xy.rename(
            columns={
                'g':self.colname_g, 
                self.colname_time:COLNAME_SURVIVAL_DURATION, 
                'true_prob':COLNAME_SURVIVAL_EVENT_OBSERVED
                }
        )
        df_xy = df_xy.loc[df_xy[self.colname_g] > 0,:]
        df_xy = df_xy.reset_index(drop=True)
        df_xy[COL_WEIGHT] = 1
        return df_xy
    

    def _get_df_Xy(self):
        if self.mode == 'first_cross_obs_only':
            df_Xy = self._get_df_Xy_trans_obs()
        elif self.mode == 'full_traj_obs_only':
            df_Xy = self._get_df_Xy_full_traj_obs()
        elif self.mode == 'true_probs_grid':
            df_Xy = self._get_df_Xy_true_prob()
        else:
            raise NotImplementedError
        df_Xy[self.colname_g] = df_Xy[self.colname_g]/5
        cols_subj_feats = sorted([i for i in df_Xy.columns if i.startswith('feat')])
        if self.separate_g_from_feats:
            subjects = df_Xy[self.colname_traj_id]
            X = torch.tensor(df_Xy[cols_subj_feats].values, dtype=torch.float32)
            g = torch.tensor(df_Xy[[self.colname_g]].values, dtype=torch.float32)
            t = torch.tensor(df_Xy[[COLNAME_SURVIVAL_DURATION]].values, dtype=torch.float32)
            y = torch.tensor(df_Xy[[COLNAME_SURVIVAL_EVENT_OBSERVED]].values, dtype=torch.float32)
            weight = torch.tensor(df_Xy[[COL_WEIGHT]].values, dtype=torch.float32)
            return subjects, X,g,t,y, weight
        
        else:
            subjects = df_Xy[self.colname_traj_id]
            cols_subj_feats = cols_subj_feats + [self.colname_g]
            X = torch.tensor(df_Xy[cols_subj_feats].values, dtype=torch.float32)
            g = None
            t = torch.tensor(df_Xy[[COLNAME_SURVIVAL_DURATION]].values, dtype=torch.float32)
            y = torch.tensor(df_Xy[[COLNAME_SURVIVAL_EVENT_OBSERVED]].values, dtype=torch.float32)
            weight = torch.tensor(df_Xy[[COL_WEIGHT]].values, dtype=torch.float32)
            return subjects,X,g,t,y, weight

class DataModuleMarkovSurvSurf(LightningDataModule):
    def __init__(
            self, 
            df_dir,
            ds_name, 
            g_resol, 
            separate_g_from_feats, 
            batch_size, 
            num_workers,
            train_mode:Literal['first_cross_obs_only','full_traj_obs_only'],
            eval_mode:Literal['first_cross_obs_only', 'true_probs_grid'], 
            t_resol=None
        ):
    
        super().__init__()
        self.df_dir = df_dir
        self.ds_name = ds_name
        self.g_resol = g_resol
        self.separate_g_from_feats = separate_g_from_feats
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_feats_g_excl = ds_name_to_n_feats_mapping[ds_name]
        self.train_mode = train_mode
        self.eval_mode = eval_mode
    def train_dataloader(self):
        train_split = DatasetMarkovSurvSurf(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='train', 
            mode=self.train_mode, 
            separate_g_from_feats=self.separate_g_from_feats)
        return DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        as_obs = DatasetMarkovSurvSurf(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='val', 
            mode=self.train_mode, 
            separate_g_from_feats=self.separate_g_from_feats
        )
        as_true_prob = DatasetMarkovSurvSurf(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='val', 
            mode=self.eval_mode, 
            separate_g_from_feats=self.separate_g_from_feats
        )

        return [
            DataLoader(as_obs, batch_size=as_obs.__len__()//20, num_workers=self.num_workers, shuffle=False), 
            DataLoader(as_true_prob, batch_size=as_true_prob.__len__()//20, num_workers=self.num_workers, shuffle=False)
        ]
    
    def test_dataloader(self):
        as_obs = DatasetMarkovSurvSurf(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='test', 
            mode=self.train_mode, 
            separate_g_from_feats=self.separate_g_from_feats
        )
        as_true_prob = DatasetMarkovSurvSurf(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='test', 
            mode=self.eval_mode, 
            separate_g_from_feats=self.separate_g_from_feats
        )

        return [
            DataLoader(as_obs, batch_size=as_obs.__len__()//20, num_workers=self.num_workers, shuffle=False), 
            DataLoader(as_true_prob, batch_size=as_true_prob.__len__()//20, num_workers=self.num_workers, shuffle=False)
        ]
