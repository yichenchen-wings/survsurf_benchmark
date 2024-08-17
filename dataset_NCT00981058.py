
from typing import Literal
from lightning import LightningDataModule
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import uniform
import pandas as pd
import os

ds_name_to_n_feats_mapping = {
    'real_NCT00981058':51,
    'real_NCT00981058_inject':52,
}

COLNAME_SURVIVAL_DURATION = "duration"
COLNAME_SURVIVAL_EVENT_OBSERVED ="event_observed"
COL_WEIGHT = 'weight'
COL_IS_TRANS = 'is_t_trans'


class DatasetNCT00981058(Dataset):
    def __init__(
            self, 
            df_dir, 
            ds_name, 
            g_resol, 
            split: Literal['train', 'tune', 'val', 'test'], 
            mode:Literal['first_cross_obs_only', 'first_last_obs_per_g', 'full_traj_obs_only', 'multi_t', 'true_probs_grid','true_probs_grid_naless'],
            separate_g_from_feats: bool,
            t_resol=1,
            g_max=5,
        ):
        self.split = split
        self.mode = mode
        self.path_df_feature_per_sub = os.path.join(df_dir,f'{ds_name}__df_features_{split}.csv')
        self.path_max_g_by_t_obs = os.path.join(df_dir,f'{ds_name}__df_state_history_sampled_max_{split}.csv')
        self.path_true_prob = os.path.join(df_dir,f'{ds_name}__df_true_prob_long_{split}.csv')
        self.g_resol = g_resol
        self.t_resol = t_resol
        self.separate_g_from_feats = separate_g_from_feats
        self.colname_traj_id = 'subject'
        self.colname_time = 't'
        self.colname_g = 'g_max_by_time'
        self.g_max = g_max
        
        self.subjects, self.X, self.g, self.t, self.y, self.weight, self.is_trans= self._get_df_Xy()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        if self.separate_g_from_feats:
            return self.subjects[index], self.X[index], self.g[index], self.t[index], self.y[index], self.weight[index], self.is_trans[index]
        else:
            return self.subjects[index], self.X[index], self.t[index], self.y[index], self.weight[index], self.is_trans[index]

    def _single_traj_to_trans_time(self, df_single_traj, higher_grade_censored=True):
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
                        COLNAME_SURVIVAL_EVENT_OBSERVED:1,
                        COLNAME_SURVIVAL_DURATION:time,
                        self.colname_g:trans_to
                    }
                )
        # at the latest obs, the more severe states 'have not yet been observed'
        if higher_grade_censored:
            rows_event_df.append(
                {
                    COLNAME_SURVIVAL_EVENT_OBSERVED:0,
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
        df_xy[COL_IS_TRANS] = 1
        return df_xy 

    def _last_obs_each_g_in_traj(self, df_single_traj):
        rows = []
        for g, df in df_single_traj.groupby(self.colname_g):
            if g > 0:
                if df.shape[0] >= 1:
                    rows.append(
                        {
                            COLNAME_SURVIVAL_EVENT_OBSERVED:1,
                            COLNAME_SURVIVAL_DURATION:df[self.colname_time].max(),
                            self.colname_g: g
                        }
                    )
            else:
                rows.append(
                    {
                        COLNAME_SURVIVAL_EVENT_OBSERVED:0,
                        COLNAME_SURVIVAL_DURATION:df[self.colname_time].max(),
                        self.colname_g: self.g_resol
                    }
                )
        
        return pd.DataFrame(rows)
        
    def _get_df_Xy_first_last_obs_per_g(self):
        xs = pd.read_csv(self.path_df_feature_per_sub, index_col=0)
        assert self.colname_traj_id in xs.columns
        assert xs[self.colname_traj_id].nunique() == xs[self.colname_traj_id].size

        max_g_by_t_obs = pd.read_csv(self.path_max_g_by_t_obs, index_col=0)
        assert self.colname_traj_id in max_g_by_t_obs.columns
        
        df_trans_time = max_g_by_t_obs.groupby(self.colname_traj_id).apply(
            lambda x: self._single_traj_to_trans_time(x, higher_grade_censored=False)
        ).reset_index(level=0)
        df_trans_time[COL_IS_TRANS] = 1

        df_last_obs = max_g_by_t_obs.groupby(self.colname_traj_id).apply(
            self._last_obs_each_g_in_traj
        ).reset_index(level=0)
        df_last_obs[COL_IS_TRANS] = 0

        df_first_last_obs = pd.concat([df_trans_time, df_last_obs])
        df_xy = df_first_last_obs.merge(xs, on=self.colname_traj_id, how='left')
        df_xy = df_xy.loc[df_xy[self.colname_g] > 0,:]
        df_xy = df_xy.reset_index(drop=True)
        n_steps_per_traj = df_xy.groupby(self.colname_traj_id).apply(lambda df: (1-df[COL_IS_TRANS]).sum())
        weight = n_steps_per_traj.mean()/n_steps_per_traj

        df_xy[COL_WEIGHT] = df_xy[self.colname_traj_id].map(weight.to_dict())
        return df_xy

    def _single_traj_full_to_label(self, df_single_traj):
        rows_event_df = pd.DataFrame()
        rows_event_df[COLNAME_SURVIVAL_DURATION] = df_single_traj[self.colname_time].values
        rows_event_df[COLNAME_SURVIVAL_EVENT_OBSERVED] = 1
        rows_event_df[self.colname_g] = df_single_traj[self.colname_g].values        
        return rows_event_df
    
    def _single_traj_g_label_trans(self, df_single_traj_g):
        out = df_single_traj_g.copy()
        selector = df_single_traj_g[COLNAME_SURVIVAL_EVENT_OBSERVED].astype(bool)
        t_min = df_single_traj_g.loc[selector, COLNAME_SURVIVAL_DURATION].min()
        selector_trans = selector & (df_single_traj_g[COLNAME_SURVIVAL_DURATION] == t_min)
        out.loc[selector_trans, COL_IS_TRANS] = 1
        out.loc[~selector_trans, COL_IS_TRANS] = 0
        return out
    
    def _get_df_Xy_full_traj_obs(self, g0_as_gres=True):
        xs = pd.read_csv(self.path_df_feature_per_sub, index_col=0)
        assert self.colname_traj_id in xs.columns
        assert xs[self.colname_traj_id].nunique() == xs[self.colname_traj_id].size

        max_g_by_t_obs = pd.read_csv(self.path_max_g_by_t_obs, index_col=0)
        assert self.colname_traj_id in max_g_by_t_obs.columns
        
        df_trans_time = max_g_by_t_obs.groupby(self.colname_traj_id).apply(
            self._single_traj_full_to_label
        ).reset_index(level=0)
        df_trans_time = df_trans_time.groupby([self.colname_traj_id, self.colname_g]).apply(
            self._single_traj_g_label_trans
        ).reset_index(drop=True)

        if g0_as_gres:
            df_trans_time.loc[
                df_trans_time[self.colname_g] == 0,
                COLNAME_SURVIVAL_EVENT_OBSERVED
            ]  = 0
            
            df_trans_time.loc[ 
                df_trans_time[self.colname_g] == 0,
                self.colname_g
            ]  = self.g_resol

            df_xy = df_trans_time.merge(xs, on=self.colname_traj_id, how='left')
            df_xy = df_xy.loc[df_xy[self.colname_g] > 0,:]
        else:
            df_xy = df_trans_time.merge(xs, on=self.colname_traj_id, how='left')
        df_xy = df_xy.reset_index(drop=True)
        df_xy[COL_WEIGHT] = 1
        return df_xy
    
    def _single_event_to_5_times(self, single_event, t_max):
        out_df = pd.DataFrame()
        assert single_event.shape[0] == 1
        single_event = single_event.iloc[0, :]
        
        out_df[COLNAME_SURVIVAL_EVENT_OBSERVED] = np.nan
        event_time = single_event[COLNAME_SURVIVAL_DURATION]
        if single_event['event_observed']:
            t = uniform.rvs(loc=0.0, scale=t_max, size=5)
            t = np.array(list(set(t).union([event_time])))
            out_df[COLNAME_SURVIVAL_DURATION] = t

            selector = out_df[COLNAME_SURVIVAL_DURATION] < 0
            out_df.loc[selector, COLNAME_SURVIVAL_DURATION] = 0

            selector = out_df[COLNAME_SURVIVAL_DURATION] < event_time
            out_df.loc[selector, COLNAME_SURVIVAL_EVENT_OBSERVED] = 0

            selector = ~selector
            out_df.loc[selector, COLNAME_SURVIVAL_EVENT_OBSERVED] = 1
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
        
        out_df[COL_WEIGHT] = 1
        assert out_df[COLNAME_SURVIVAL_EVENT_OBSERVED].notna().all()
        assert out_df[COL_WEIGHT].notna().all()
        return out_df
        
    def _get_df_Xy_multi_t(self):
        xs = pd.read_csv(self.path_df_feature_per_sub, index_col=0)
        assert self.colname_traj_id in xs.columns
        assert xs[self.colname_traj_id].nunique() == xs[self.colname_traj_id].size

        max_g_by_t_obs = pd.read_csv(self.path_max_g_by_t_obs, index_col=0)
        df_trans_time = max_g_by_t_obs.groupby(self.colname_traj_id).apply(
            self._single_traj_to_trans_time
        ).reset_index(level=0)
        t_max = df_trans_time[COLNAME_SURVIVAL_DURATION].max()
        df_trans_time = df_trans_time.groupby(
            [self.colname_traj_id, self.colname_g],
            group_keys=True
        ).apply(
            lambda df: self._single_event_to_5_times(df, t_max=t_max)
        ).reset_index(level=[0,1])
        df_xy = df_trans_time.merge(xs, on=self.colname_traj_id, how='left')
        df_xy = df_xy.reset_index(drop=True)
        df_xy[COL_IS_TRANS] = np.nan
        return df_xy 
     
    def _single_subj_to_tg_grid(self, obs_full_traj_subj, t_min, t_max):
        ts = [i for i in range(int(t_min), int(t_max)+1, self.t_resol)]
        gs = [i+1 for i in range(self.g_max)]
        ts = np.array(sorted(set(ts)))

        rows = []
        obs_full_traj_subj = obs_full_traj_subj.sort_values(COLNAME_SURVIVAL_DURATION)
        obs_full_traj_subj[COLNAME_SURVIVAL_EVENT_OBSERVED] = obs_full_traj_subj[COLNAME_SURVIVAL_EVENT_OBSERVED].astype(bool)
        t_max = obs_full_traj_subj[COLNAME_SURVIVAL_DURATION].max()
        for g in gs:
            if g in obs_full_traj_subj[self.colname_g].values:
                obs_trans_time_g = obs_full_traj_subj.loc[obs_full_traj_subj[self.colname_g] == g, :]
                obs_trans_time_g.shape[0] >= 1
                obs_trans_time_g = obs_trans_time_g.iloc[0,:]
             
                event_observed = obs_trans_time_g[COLNAME_SURVIVAL_EVENT_OBSERVED]
                t_observed_or_censored = obs_trans_time_g[COLNAME_SURVIVAL_DURATION]
                t_uncensored_alive = np.nan,
                missing_exact_t = False
   
            else:
                obs_trans_time_g_greater = obs_full_traj_subj.loc[
                    (obs_full_traj_subj[self.colname_g] > g) & obs_full_traj_subj[COLNAME_SURVIVAL_EVENT_OBSERVED], :
                ]
                if obs_trans_time_g_greater.shape[0] >= 1:
                    obs_trans_time_g_greater = obs_trans_time_g_greater.iloc[0,:]

                    event_observed = True
                    t_observed_or_censored = obs_trans_time_g_greater[COLNAME_SURVIVAL_DURATION]
                    missing_exact_t = True

                    obs_trans_time_g_smaller = obs_full_traj_subj.loc[
                        (obs_full_traj_subj[self.colname_g] < g) & obs_full_traj_subj[COLNAME_SURVIVAL_EVENT_OBSERVED], :
                    ]
                    if obs_trans_time_g_smaller.shape[0] >= 1:
                        obs_trans_time_g_smaller = obs_trans_time_g_smaller.iloc[-1,:]
                        t_uncensored_alive = obs_trans_time_g_smaller[COLNAME_SURVIVAL_DURATION]
                    else:
                        t_uncensored_alive = 0
                else:
                    event_observed = False
                    t_observed_or_censored = t_max
                    t_uncensored_alive = np.nan
                    missing_exact_t = False

            df = pd.DataFrame()
            df[COLNAME_SURVIVAL_DURATION] = ts
            df[self.colname_g] = g
            df[COLNAME_SURVIVAL_EVENT_OBSERVED] = np.nan
            df[COL_WEIGHT] = 1
            if event_observed:
                if missing_exact_t:
                    selector = ts >= t_observed_or_censored
                    df.loc[selector, COLNAME_SURVIVAL_EVENT_OBSERVED] = 1
                    
                    selector = ts <= t_uncensored_alive
                    df.loc[selector, COLNAME_SURVIVAL_EVENT_OBSERVED] = 0

                else:
                    selector = ts >= t_observed_or_censored
                    df.loc[selector, COLNAME_SURVIVAL_EVENT_OBSERVED] = 1
                    df.loc[~selector, COLNAME_SURVIVAL_EVENT_OBSERVED] = 0
            else:
                selector = ts <= t_observed_or_censored
                df.loc[selector, COLNAME_SURVIVAL_EVENT_OBSERVED] = 0
            rows.append(df)
        return pd.concat(rows)
    
    def _get_df_Xy_true_prob(self, dropna=False):
        xs = pd.read_csv(self.path_df_feature_per_sub, index_col=0)
        assert self.colname_traj_id in xs.columns
        assert xs[self.colname_traj_id].nunique() == xs[self.colname_traj_id].size

        max_g_by_t_obs = pd.read_csv(self.path_max_g_by_t_obs, index_col=0)
        df_full_traj = max_g_by_t_obs.groupby(self.colname_traj_id).apply(
            self._single_traj_full_to_label
        ).reset_index(level=0)
        t_min = max_g_by_t_obs[self.colname_time].min()
        t_max = max_g_by_t_obs[self.colname_time].max()
        df_tg_grid = df_full_traj.groupby(
            self.colname_traj_id,
            group_keys=True
        ).apply(
            lambda df: self._single_subj_to_tg_grid(df, t_min, t_max)
        ).reset_index(level=[0])
        
        if dropna:
            df_tg_grid = df_tg_grid.dropna()
        df_xy = df_tg_grid.merge(xs, on=self.colname_traj_id, how='left')
        df_xy = df_xy.reset_index(drop=True)
        df_xy[COL_IS_TRANS] = np.nan
        return df_xy

    def _get_df_Xy(self):
        if self.mode == 'first_cross_obs_only':
            df_Xy = self._get_df_Xy_trans_obs()
        elif self.mode == 'first_last_obs_per_g':
            df_Xy = self._get_df_Xy_first_last_obs_per_g()
        elif self.mode == 'full_traj_obs_only':
            df_Xy = self._get_df_Xy_full_traj_obs()
        elif self.mode == 'true_probs_grid':
            df_Xy = self._get_df_Xy_true_prob()
        elif self.mode == 'true_probs_grid_naless':
            df_Xy = self._get_df_Xy_true_prob(dropna=True)
        elif self.mode == 'multi_t':
            df_Xy = self._get_df_Xy_multi_t()
        else:
            raise NotImplementedError
        df_Xy[self.colname_g] = df_Xy[self.colname_g]/self.g_max
        cols_subj_feats = sorted([i for i in df_Xy.columns if i.startswith('feat')])
        if self.separate_g_from_feats:
            subjects = df_Xy[self.colname_traj_id]
            X = torch.tensor(df_Xy[cols_subj_feats].values, dtype=torch.float32)
            g = torch.tensor(df_Xy[[self.colname_g]].values, dtype=torch.float32)
            t = torch.tensor(df_Xy[[COLNAME_SURVIVAL_DURATION]].values, dtype=torch.float32)
            y = torch.tensor(df_Xy[[COLNAME_SURVIVAL_EVENT_OBSERVED]].values, dtype=torch.float32)
            weight = torch.tensor(df_Xy[[COL_WEIGHT]].values, dtype=torch.float32)
            is_trans = torch.tensor(df_Xy[[COL_IS_TRANS]].values, dtype=torch.float32)
            return subjects, X,g,t,y, weight, is_trans
        
        else:
            subjects = df_Xy[self.colname_traj_id]
            cols_subj_feats = cols_subj_feats + [self.colname_g]
            X = torch.tensor(df_Xy[cols_subj_feats].values, dtype=torch.float32)
            g = None
            t = torch.tensor(df_Xy[[COLNAME_SURVIVAL_DURATION]].values, dtype=torch.float32)
            y = torch.tensor(df_Xy[[COLNAME_SURVIVAL_EVENT_OBSERVED]].values, dtype=torch.float32)
            weight = torch.tensor(df_Xy[[COL_WEIGHT]].values, dtype=torch.float32)
            is_trans = torch.tensor(df_Xy[[COL_IS_TRANS]].values, dtype=torch.float32)
            return subjects,X,g,t,y, weight, is_trans

class DataModuleNCT00981058(LightningDataModule):
    def __init__(
            self, 
            df_dir,
            ds_name, 
            g_resol, 
            separate_g_from_feats, 
            batch_size, 
            num_workers,
            train_mode:Literal['first_cross_obs_only','full_traj_obs_only'],
            eval_mode:Literal['first_cross_obs_only', 'true_probs_grid', 'multi_t','true_probs_grid_naless'], 
            t_resol=None,
            g_max=5
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
        self.g_max = g_max
    def train_dataloader(self):
        train_split = DatasetNCT00981058(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='train', 
            mode=self.train_mode, 
            separate_g_from_feats=self.separate_g_from_feats,
            t_resol=self.t_resol,
            g_max=self.g_max
        )
        return DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        as_obs = DatasetNCT00981058(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='val', 
            mode=self.train_mode, 
            separate_g_from_feats=self.separate_g_from_feats,
            t_resol=self.t_resol,
            g_max=self.g_max
        )
        as_true_prob = DatasetNCT00981058(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='val', 
            mode=self.eval_mode, 
            separate_g_from_feats=self.separate_g_from_feats,
            t_resol=self.t_resol,
            g_max=self.g_max
        )

        return [
            DataLoader(as_obs, batch_size=as_obs.__len__()//20, num_workers=self.num_workers, shuffle=False), 
            DataLoader(as_true_prob, batch_size=as_true_prob.__len__()//20, num_workers=self.num_workers, shuffle=False)
        ]
    
    def test_dataloader(self):
        as_obs = DatasetNCT00981058(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='test', 
            mode=self.train_mode, 
            separate_g_from_feats=self.separate_g_from_feats,
            t_resol=self.t_resol,
            g_max=self.g_max
        )
        as_true_prob = DatasetNCT00981058(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='test', 
            mode=self.eval_mode, 
            separate_g_from_feats=self.separate_g_from_feats,
            t_resol=self.t_resol,
            g_max=self.g_max
        )

        return [
            DataLoader(as_obs, batch_size=as_obs.__len__()//20, num_workers=self.num_workers, shuffle=False), 
            DataLoader(as_true_prob, batch_size=as_true_prob.__len__()//20, num_workers=self.num_workers, shuffle=False)
        ]
    
    
    def test_mode_train_dataloader(self):
        as_obs = DatasetNCT00981058(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='train', 
            mode=self.train_mode, 
            separate_g_from_feats=self.separate_g_from_feats,
            t_resol=self.t_resol,
            g_max=self.g_max
        )
        as_true_prob = DatasetNCT00981058(
            self.df_dir, 
            self.ds_name, 
            self.g_resol, 
            split='train', 
            mode=self.eval_mode, 
            separate_g_from_feats=self.separate_g_from_feats,
            t_resol=self.t_resol,
            g_max=self.g_max
        )

        return [
            DataLoader(as_obs, batch_size=as_obs.__len__()//20, num_workers=self.num_workers, shuffle=False), 
            DataLoader(as_true_prob, batch_size=as_true_prob.__len__()//20, num_workers=self.num_workers, shuffle=False)
        ]
