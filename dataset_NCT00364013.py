
from lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Literal
from scipy.stats import uniform
from sksurv.nonparametric import kaplan_meier_estimator
import pandas as pd
import numpy as np
import os

from IPCW import IPCW

ds_name_to_n_feats_mapping = {
    'real_NCT00364013':25,
}

COLNAME_SURVIVAL_DURATION = "duration"
COLNAME_SURVIVAL_EVENT_OBSERVED ="event_observed"
COLNAME_WEIGHT = 'weight'

def get_train_ipcw(df_dir, ds_name):
    path_event_g_at_t_obs = os.path.join(df_dir,f'{ds_name}__df_event_time_train.csv')
    df_event_g_at_t_obs = pd.read_csv(path_event_g_at_t_obs, index_col=0)
    g_to_censor_curv = dict()
    for g, df_event_t in df_event_g_at_t_obs.groupby('g'):
        df_event_t = df_event_t.sort_values('t')
        time, S_censor = kaplan_meier_estimator(df_event_t['event_observed'].astype(bool), df_event_t['t'], reverse=True)
        if 0 not in time:
            time = np.r_[[0.], time]
            S_censor = np.r_[[1.], S_censor]
        g_to_censor_curv[g] = IPCW(time, S_censor)
    return g_to_censor_curv


def get_t_max_train(df_dir, ds_name):
    path_event_g_at_t_obs = os.path.join(df_dir,f'{ds_name}__df_event_time_train.csv')
    df_event_g_at_t_obs = pd.read_csv(path_event_g_at_t_obs, index_col=0)
    g_to_t_max = dict()
    for g, df_event_t in df_event_g_at_t_obs.groupby('g'):
        df_event_t = df_event_t.sort_values('t')
        g_to_t_max[g] = df_event_t['t'].max()
    return g_to_t_max



class DatasetNCT00364013SurvCurv(Dataset):
    def __init__(
            self, 
            df_dir, 
            ds_name, 
            g_resol, 
            split: Literal['train', 'tune', 'val', 'test'], 
            mode:Literal['obs_only','obs_only_traj', 'multi_t', 'true_probs_grid', 'true_probs_around_t'],
            separate_g_from_feats: bool,
            t_resol=30
        ):
        self.split = split
        self.path_df_feature_per_sub = os.path.join(df_dir,f'{ds_name}__df_features_{split}.csv')
        self.path_event_g_at_t_obs = os.path.join(df_dir,f'{ds_name}__df_event_time_{split}.csv')
        self.g_resol = g_resol
        self.t_resol = t_resol
        self.separate_g_from_feats = separate_g_from_feats
        self.colname_traj_id = 'subject'
        self.colname_time = 't'
        self.colname_trans_to = 'g'
        self.mode = mode
        self.ipcw_dict = get_train_ipcw(df_dir=df_dir, ds_name=ds_name)
        self.g_to_t_max =  get_t_max_train(df_dir=df_dir, ds_name=ds_name)

        self.subjects, self.X, self.g, self.t, self.y, self.weight= self._get_df_Xy()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        if self.separate_g_from_feats:
            return self.subjects[index], self.X[index], self.g[index], self.t[index], self.y[index], self.weight[index]
        else:
            return self.subjects[index], self.X[index], self.t[index], self.y[index], self.weight[index]
        
    def _single_event_to_5_times(self, single_event):

        out_df = pd.DataFrame()
        assert single_event.shape[0] == 1
        single_event = single_event.iloc[0, :]
        
        out_df[COLNAME_SURVIVAL_EVENT_OBSERVED] = np.nan
        out_df[COLNAME_WEIGHT] = np.nan
        event_time = single_event[self.colname_time]
        g = single_event['g']
        ipcw = self.ipcw_dict[g]
        t_max = self.g_to_t_max[g]
        if single_event['event_observed']:
            t = uniform.rvs(loc=0.0, scale=t_max, size=5)
            t = np.array(list(set(t).union([event_time])))
            out_df[COLNAME_SURVIVAL_DURATION] = t

            selector = out_df[COLNAME_SURVIVAL_DURATION] < 0
            out_df.loc[selector, COLNAME_SURVIVAL_DURATION] = 0

            selector = out_df[COLNAME_SURVIVAL_DURATION] < event_time
            out_df.loc[selector, COLNAME_SURVIVAL_EVENT_OBSERVED] = 0
            out_df.loc[selector, COLNAME_WEIGHT] = ipcw(out_df.loc[selector, COLNAME_SURVIVAL_DURATION])

            selector = ~selector
            out_df.loc[selector, COLNAME_SURVIVAL_EVENT_OBSERVED] = 1
            out_df.loc[selector, COLNAME_WEIGHT] = ipcw(event_time)
        else:
            if event_time:
                t = uniform.rvs(loc=0.0, scale=event_time, size=5)
                t = np.array(list(set(t).union([event_time])))
            else:
                t = 0
            out_df[COLNAME_SURVIVAL_DURATION] = t

            selector = out_df[COLNAME_SURVIVAL_DURATION] < 0
            out_df.loc[selector, COLNAME_SURVIVAL_DURATION] = 0
            out_df.loc[
                out_df[COLNAME_SURVIVAL_DURATION] <= event_time, 
                COLNAME_SURVIVAL_EVENT_OBSERVED
            ] = 0
            out_df[COLNAME_WEIGHT] = ipcw(out_df[COLNAME_SURVIVAL_DURATION])
        
        assert out_df[COLNAME_SURVIVAL_EVENT_OBSERVED].notna().all()
        assert out_df[COLNAME_WEIGHT].notna().all()
        return out_df
        

    def _get_df_Xy_multi_t(self):
        xs = pd.read_csv(self.path_df_feature_per_sub, index_col=0)
        assert self.colname_traj_id in xs.columns
        assert xs[self.colname_traj_id].nunique() == xs[self.colname_traj_id].size

        event_g_at_t_obs = pd.read_csv(self.path_event_g_at_t_obs, index_col=0)
        df_trans_time = event_g_at_t_obs.groupby(
            [self.colname_traj_id,'g'],
            group_keys=True
        ).apply(
            lambda df: self._single_event_to_5_times(df)
        ).reset_index(level=[0,1]).rename(
            columns={
                'g':self.colname_trans_to
            }
        )
        df_xy = df_trans_time.merge(xs, on=self.colname_traj_id, how='left')
        df_xy = df_xy.reset_index(drop=True)
        return df_xy 
    
    def _single_traj_from_obs(self, df):
        assert df.shape[0] == 2
        df = df.sort_values('g')
        assert all(df['g'].values == np.array([1,2]))

        both_observed = all(df['event_observed'] == 1)
        one_observed = df['event_observed'].sum() == 1
        both_equal = df[self.colname_time].nunique() == 1

        
        t_g1_start = df.loc[df['g'] == 1, self.colname_time].iloc[0]
        t_g2_start = df.loc[df['g'] == 2, self.colname_time].iloc[0]

        out_df = []
        if both_observed and both_equal:
            entry = {
                self.colname_trans_to: 2,
                COLNAME_SURVIVAL_DURATION: t_g1_start,
                COLNAME_SURVIVAL_EVENT_OBSERVED:1,
            }
            out_df.append(entry)
        elif both_observed:
            t_g1_end = t_g2_start - self.g_resol
            t_g1_end = max((t_g2_start+t_g1_start)/2, t_g1_end)
            entry = {
                self.colname_trans_to: 1,
                COLNAME_SURVIVAL_DURATION: t_g1_start,
                COLNAME_SURVIVAL_EVENT_OBSERVED:1,
            }
            out_df.append(entry)

            entry = {
                self.colname_trans_to: 1,
                COLNAME_SURVIVAL_DURATION: t_g1_end,
                COLNAME_SURVIVAL_EVENT_OBSERVED:1
            }
            out_df.append(entry)
            
            entry = {
                self.colname_trans_to: 2,
                COLNAME_SURVIVAL_DURATION: t_g2_start,
                COLNAME_SURVIVAL_EVENT_OBSERVED:1
            }
            out_df.append(entry)
        elif one_observed:
            assert all(df.loc[df['g'] == 1, 'event_observed'] == 1)
            entry = {
                self.colname_trans_to: 1,
                COLNAME_SURVIVAL_DURATION: t_g1_start,
                COLNAME_SURVIVAL_EVENT_OBSERVED:1
            }
            out_df.append(entry)
            
            entry = {
                self.colname_trans_to: 1,
                COLNAME_SURVIVAL_DURATION: t_g2_start,
                COLNAME_SURVIVAL_EVENT_OBSERVED:1
            }
            out_df.append(entry)
        else:
            entry = {
                self.colname_trans_to: 1,
                COLNAME_SURVIVAL_DURATION: t_g1_start,
                COLNAME_SURVIVAL_EVENT_OBSERVED:0
            }
            out_df.append(entry)
            
            entry = {
                self.colname_trans_to: 2,
                COLNAME_SURVIVAL_DURATION: t_g2_start,
                COLNAME_SURVIVAL_EVENT_OBSERVED:0
            }
            out_df.append(entry)
        out_df = pd.DataFrame(out_df)
        out_df[COLNAME_WEIGHT] = 1
        return out_df

    def _get_df_Xy_obs_traj(self):
        xs = pd.read_csv(self.path_df_feature_per_sub, index_col=0)
        assert self.colname_traj_id in xs.columns
        assert xs[self.colname_traj_id].nunique() == xs[self.colname_traj_id].size

        event_g_at_t_obs = pd.read_csv(self.path_event_g_at_t_obs, index_col=0)
        df_trans_time = event_g_at_t_obs.groupby(
            [self.colname_traj_id],
            group_keys=True
        ).apply(
            lambda df: self._single_traj_from_obs(df)
        ).reset_index(level=[0]).rename(
            columns={
                'g':self.colname_trans_to
            }
        )
        df_xy = df_trans_time.merge(xs, on=self.colname_traj_id, how='left')
        df_xy = df_xy.reset_index(drop=True)
        return df_xy


    def _get_df_Xy_obs(self):
        xs = pd.read_csv(self.path_df_feature_per_sub, index_col=0)
        assert self.colname_traj_id in xs.columns
        assert xs[self.colname_traj_id].nunique() == xs[self.colname_traj_id].size

        event_g_at_t_obs = pd.read_csv(self.path_event_g_at_t_obs, index_col=0)
        assert self.colname_traj_id in event_g_at_t_obs.columns
        for g in event_g_at_t_obs['g'].unique():
            selector = event_g_at_t_obs['g'] == g
            ipcw = self.ipcw_dict[g]
            event_g_at_t_obs.loc[selector, COLNAME_WEIGHT] = ipcw(
                event_g_at_t_obs.loc[selector,self.colname_time]
            )
        
        df_trans_time = event_g_at_t_obs.rename(
            columns={
                self.colname_time:COLNAME_SURVIVAL_DURATION,
                'g':self.colname_trans_to, 
                'event_observed':COLNAME_SURVIVAL_EVENT_OBSERVED
            }
        )
        df_xy = df_trans_time.merge(xs, on=self.colname_traj_id, how='left')
        df_xy = df_xy.reset_index(drop=True)
        assert df_xy.notna().all().all()
        return df_xy
     
    def _get_df_Xy_around_t(self): # entries include event time and t_res before the event time
        xs = pd.read_csv(self.path_df_feature_per_sub, index_col=0)
        assert self.colname_traj_id in xs.columns
        assert xs[self.colname_traj_id].nunique() == xs[self.colname_traj_id].size

        event_g_at_t_obs = pd.read_csv(self.path_event_g_at_t_obs, index_col=0)
        assert self.colname_traj_id in event_g_at_t_obs.columns

        event_g_at_t_obs_before = event_g_at_t_obs.copy()
        event_g_at_t_obs_before[self.colname_time] = event_g_at_t_obs_before[self.colname_time] - self.t_resol
        selector = event_g_at_t_obs_before[self.colname_time] < 0
        event_g_at_t_obs_before.loc[selector, self.colname_time] = 0
        event_g_at_t_obs_before['event_observed'] = 0
        
        df_trans_time = pd.concat([event_g_at_t_obs, event_g_at_t_obs_before]).rename(
            columns={
                self.colname_time:COLNAME_SURVIVAL_DURATION,
                'g':self.colname_trans_to, 
                'event_observed':COLNAME_SURVIVAL_EVENT_OBSERVED
            }
        )
    
        for g in df_trans_time['g'].unique():
            selector = df_trans_time['g'] == g
            ipcw = self.ipcw_dict[g]
            df_trans_time.loc[selector, COLNAME_WEIGHT] = ipcw(
                df_trans_time.loc[selector,COLNAME_SURVIVAL_DURATION]
            )
        df_xy = df_trans_time.merge(xs, on=self.colname_traj_id, how='left')
        df_xy = df_xy.reset_index(drop=True)
        return df_xy  

    def _single_event_to_grid_times(self, single_event, t_min, t_max):
        t = [i for i in range(int(t_min), int(t_max)+1, self.t_resol)] + [t_max]
        t = sorted(set(t))

        out_df = pd.DataFrame()
        assert single_event.shape[0] == 1
        single_event = single_event.iloc[0, :]
        out_df[COLNAME_SURVIVAL_DURATION] = t
        out_df[COLNAME_SURVIVAL_EVENT_OBSERVED] = np.nan
        if single_event['event_observed']:
            out_df.loc[
                out_df[COLNAME_SURVIVAL_DURATION] < single_event[self.colname_time], 
                COLNAME_SURVIVAL_EVENT_OBSERVED
            ] = 0
            out_df.loc[
                out_df[COLNAME_SURVIVAL_DURATION] >= single_event[self.colname_time], 
                COLNAME_SURVIVAL_EVENT_OBSERVED
            ] = 1
        else:
            out_df.loc[
                out_df[COLNAME_SURVIVAL_DURATION] <= single_event[self.colname_time], 
                COLNAME_SURVIVAL_EVENT_OBSERVED
            ] = 0
        out_df[COLNAME_WEIGHT] = 1
        return out_df
    
    def _get_df_Xy_true_prob(self):
        xs = pd.read_csv(self.path_df_feature_per_sub, index_col=0)
        assert self.colname_traj_id in xs.columns
        assert xs[self.colname_traj_id].nunique() == xs[self.colname_traj_id].size

        event_g_at_t_obs = pd.read_csv(self.path_event_g_at_t_obs, index_col=0)
        t_min = event_g_at_t_obs.loc[~event_g_at_t_obs['event_observed'].astype(bool),self.colname_time].min()
        t_max = event_g_at_t_obs.loc[event_g_at_t_obs['event_observed'].astype(bool),self.colname_time].max()
        df_trans_time = event_g_at_t_obs.groupby(
            [self.colname_traj_id,'g'],
            group_keys=True
        ).apply(
            lambda df: self._single_event_to_grid_times(df, t_min, t_max)
        ).reset_index(level=[0,1]).rename(
            columns={
                'g':self.colname_trans_to
            }
        )
        df_xy = df_trans_time.merge(xs, on=self.colname_traj_id, how='left')
        df_xy = df_xy.reset_index(drop=True)
        return df_xy

    def _get_df_Xy(self):
        if self.mode == 'obs_only':
            df_Xy = self._get_df_Xy_obs()
        elif self.mode == 'obs_only_traj':
            df_Xy = self._get_df_Xy_obs_traj()
        elif self.mode == 'true_probs_grid':
            df_Xy = self._get_df_Xy_true_prob()
        elif self.mode == 'true_probs_around_t':
            df_Xy = self._get_df_Xy_around_t()
        elif self.mode == 'multi_t':
            df_Xy = self._get_df_Xy_multi_t()
        else:
            raise NotImplementedError
        df_Xy[COLNAME_WEIGHT] = df_Xy[COLNAME_WEIGHT]/df_Xy[COLNAME_WEIGHT].sum() * df_Xy.shape[0]
        df_Xy[self.colname_trans_to] = df_Xy[self.colname_trans_to]/2
        cols_subj_feats = sorted([i for i in df_Xy.columns if i.startswith('feat')])
        if self.separate_g_from_feats:
            subjects = df_Xy[self.colname_traj_id]
            X = torch.tensor(df_Xy[cols_subj_feats].values.astype(np.float64), dtype=torch.float32)
            g = torch.tensor(df_Xy[[self.colname_trans_to]].values.astype(np.float64), dtype=torch.float32)
            t = torch.tensor(df_Xy[[COLNAME_SURVIVAL_DURATION]].values.astype(np.float64), dtype=torch.float32)
            y = torch.tensor(df_Xy[[COLNAME_SURVIVAL_EVENT_OBSERVED]].values.astype(np.float64), dtype=torch.float32)
            weight = torch.tensor(df_Xy[[COLNAME_WEIGHT]].values.astype(np.float64), dtype=torch.float32)
            return subjects, X,g,t,y, weight
        
        else:
            subjects = df_Xy[self.colname_traj_id]
            cols_subj_feats = cols_subj_feats + [self.colname_trans_to]
            X = torch.tensor(df_Xy[cols_subj_feats].values.astype(np.float64), dtype=torch.float32)
            g = None
            t = torch.tensor(df_Xy[[COLNAME_SURVIVAL_DURATION]].values.astype(np.float64), dtype=torch.float32)
            y = torch.tensor(df_Xy[[COLNAME_SURVIVAL_EVENT_OBSERVED]].values.astype(np.float64), dtype=torch.float32)
            weight = torch.tensor(df_Xy[[COLNAME_WEIGHT]].values.astype(np.float64), dtype=torch.float32)
            return subjects,X,g,t,y, weight

class DataModuleNCT00364013SurvCurv(LightningDataModule):
    def __init__(
            self, 
            df_dir, 
            ds_name, 
            g_resol, 
            separate_g_from_feats, 
            batch_size, 
            num_workers, 
            train_mode:Literal['obs_only', 'multi_t', 'true_probs_grid', 'true_probs_around_t'],
            eval_mode:Literal['obs_only', 'multi_t', 'true_probs_grid', 'true_probs_around_t'], 
            t_resol=30
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
        self.t_resol = t_resol
    def train_dataloader(self):
        train_split = DatasetNCT00364013SurvCurv(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='train', 
            mode=self.train_mode,
            separate_g_from_feats=self.separate_g_from_feats,
            t_resol=self.t_resol
        )
        return DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    def train_for_eval_dataloader(self):
        as_obs = DatasetNCT00364013SurvCurv(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='train', 
            mode=self.train_mode,
            separate_g_from_feats=self.separate_g_from_feats,
            t_resol=self.t_resol
        )
        as_true_prob = DatasetNCT00364013SurvCurv(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='train', 
            mode=self.eval_mode, 
            separate_g_from_feats=self.separate_g_from_feats,
            t_resol=self.t_resol
        )
        return [
            DataLoader(as_obs, batch_size=as_obs.__len__()//20, num_workers=self.num_workers, shuffle=False), 
            DataLoader(as_true_prob, batch_size=as_obs.__len__()//20, num_workers=self.num_workers, shuffle=False), 
        ]
    def val_dataloader(self):
        as_obs = DatasetNCT00364013SurvCurv(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='val', 
            mode=self.train_mode, 
            separate_g_from_feats=self.separate_g_from_feats,
            t_resol=self.t_resol
        )
        as_true_prob = DatasetNCT00364013SurvCurv(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='val', 
            mode=self.eval_mode, 
            separate_g_from_feats=self.separate_g_from_feats,
            t_resol=self.t_resol
        )
        return [
            DataLoader(as_obs, batch_size=as_obs.__len__()//20, num_workers=self.num_workers, shuffle=False), 
            DataLoader(as_true_prob, batch_size=as_obs.__len__()//20, num_workers=self.num_workers, shuffle=False), 
        ]
    def test_dataloader(self):
        as_obs = DatasetNCT00364013SurvCurv(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='test', 
            mode=self.train_mode, 
            separate_g_from_feats=self.separate_g_from_feats,
            t_resol=self.t_resol
        )
        as_true_prob = DatasetNCT00364013SurvCurv(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='test', 
            mode=self.eval_mode, 
            separate_g_from_feats=self.separate_g_from_feats,
            t_resol=self.t_resol
        )
       
        return [
            DataLoader(as_obs, batch_size=as_obs.__len__()//20, num_workers=self.num_workers, shuffle=False), 
            DataLoader(as_true_prob, batch_size=as_obs.__len__()//20, num_workers=self.num_workers, shuffle=False), 
        ]

        