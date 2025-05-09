from typing import Literal
import numpy as np
import pandas as pd

COLNAME_SURVIVAL_DURATION = "duration"
COLNAME_SURVIVAL_EVENT_OBSERVED ="event_observed"

def _get_last_obs_time(df_max_g_by_t_subj, g):
    seletor_max_g = df_max_g_by_t_subj['g_max_by_time'] >= g
    t_last_obs = df_max_g_by_t_subj['t'].max()
    obs = seletor_max_g.any()
    record = dict()
    record['g'] = g
    record[COLNAME_SURVIVAL_DURATION] = t_last_obs
    record[COLNAME_SURVIVAL_EVENT_OBSERVED] = obs 
    return pd.Series(record)

from sksurv.nonparametric import CensoringDistributionEstimator


def get_brier_and_auc(out_model, df_last_obs_time_train, ipcw:Literal['by_grade', 'by_subj'], max_grade=5, max_time=None):
    integrated_brier_events = []
    mean_auc_events = []
    out_model_obs = out_model['obs'].copy()
    gs_all = out_model_obs['g'].unique()

    def fill_in_higher_gs(df):
        subj = df['subj'].iloc[0]
        max_time = df['t'].max()
        max_g = df['g'].max()
        gs_higher = gs_all[gs_all > max_g]
        if gs_higher.size:
            gs_higher_df = pd.DataFrame(columns=df.columns)
            gs_higher_df['g'] = gs_higher
            gs_higher_df['t'] = max_time
            gs_higher_df['truth'] = 0
            gs_higher_df['subj'] = subj
            return pd.concat([df, gs_higher_df])
        else:
            return df
    out_model_obs = out_model_obs.groupby('subj').apply(fill_in_higher_gs).reset_index(drop=True)   
    max_grade_in_data = df_last_obs_time_train['g'].max()
    for g, out_model_obs_sub in out_model_obs.groupby('g'):
        if g > max_grade:
            continue
        if ipcw == 'by_grade':
            df_train_obs_sub = df_last_obs_time_train.loc[
                df_last_obs_time_train['g'] == g, 
                [COLNAME_SURVIVAL_EVENT_OBSERVED, COLNAME_SURVIVAL_DURATION]
            ].copy()
        elif ipcw == 'by_subj':
            df_train_obs_sub = df_last_obs_time_train.loc[
                df_last_obs_time_train['g'] == max_grade_in_data, 
                [COLNAME_SURVIVAL_EVENT_OBSERVED, COLNAME_SURVIVAL_DURATION]
            ].copy()
            df_train_obs_sub[COLNAME_SURVIVAL_EVENT_OBSERVED] = False
        else:
            raise NotImplementedError

        if not max_time: 
            max_surv_time_train = df_last_obs_time_train.loc[
                df_last_obs_time_train[COLNAME_SURVIVAL_EVENT_OBSERVED] & (df_last_obs_time_train['g'] == g), 
                COLNAME_SURVIVAL_DURATION
            ].max()
        else:
            max_surv_time_train = max_time
        df_val_estimate = out_model_obs_sub[
            ['subj','g']
        ].merge(
            out_model['true_prob'], 
            how='left', 
            on=['subj', 'g']
        )

        df_val_survival = df_val_estimate[['subj','g']].drop_duplicates().merge(
            out_model_obs, 
            on=['subj', 'g'],
            how='left', 
        )
        df_val_survival = df_val_survival.sort_values(['subj','g']).set_index(['subj','g'])
        
        t_max = max_surv_time_train
        df_val_survival = df_val_survival.loc[df_val_survival['t'] < t_max] #may remove subjects whose event time is later than the latest censoring in training time
        t_max = df_val_survival.loc[~df_val_survival['truth'].astype(bool),'t'].max()
        t_min= df_val_survival.loc[df_val_survival['truth'].astype(bool),'t'].min()
        df_val_estimate = df_val_estimate.loc[
            (df_val_estimate['t'] < t_max) & (df_val_estimate['t'] >= t_min),
            :
        ]

        df_val_estimate_pivot = 1-df_val_estimate.set_index(['subj', 'g']).pivot(values='pred', columns='t')# Prob of still alive (i.e. 1 - prob(event happens))
        df_val_estimate_pivot = df_val_estimate_pivot.sort_index()
        subj_g_common = [i for i in df_val_estimate_pivot.index if i in df_val_survival.index]
        
        df_val_survival = df_val_survival.loc[subj_g_common,:]
        df_val_estimate_pivot = df_val_estimate_pivot.loc[subj_g_common,:]

        assert all(np.diff(df_val_estimate_pivot.columns.values) > 0)

        assert all(df_val_estimate_pivot.index.values == df_val_survival.index.values)
        df_val_survival = df_val_survival.reset_index(drop=False)

        from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc

        survival_train = df_train_obs_sub.copy()
        survival_train = survival_train.sort_values(COLNAME_SURVIVAL_DURATION)
        survival_train[COLNAME_SURVIVAL_EVENT_OBSERVED].iloc[0] = 1 # otherwise CensoringDistributionEstimator in integrated_brier_score won't run
        survival_train[COLNAME_SURVIVAL_EVENT_OBSERVED] = survival_train[COLNAME_SURVIVAL_EVENT_OBSERVED].astype(bool)
        survival_train = survival_train.to_records(index=False)

        survival_test = df_val_survival[['truth', 't']].copy()
        survival_test['truth'] = survival_test['truth'].astype(bool)
        survival_test = survival_test.to_records(index=False)

        assert survival_test.shape[0] >= 1, f'{survival_test.shape}, {g}'
        assert survival_train.shape[0] >= 1, f'{survival_test.shape}, {g}'

        intrg_brier = integrated_brier_score(
            survival_train=survival_train,
            survival_test=survival_test,
            estimate=df_val_estimate_pivot.values,
            times=df_val_estimate_pivot.columns.values
        )

        auc, mean_auc = cumulative_dynamic_auc(
            survival_train=survival_train,
            survival_test=survival_test,
            estimate=1-df_val_estimate_pivot.values,
            times=df_val_estimate_pivot.columns.values
        )
        mean_auc = np.nan
        integrated_brier_events.append(intrg_brier)
        mean_auc_events.append(mean_auc)
    return np.mean(integrated_brier_events), np.mean(mean_auc_events)


def _all_grades_start_end_time_subj(obs_full_traj_subj, gs):
    rows = []
    obs_full_traj_subj = obs_full_traj_subj.sort_values('duration')
    obs_full_traj_subj['event_observed'] = obs_full_traj_subj['event_observed'].astype(bool)
    t_max = obs_full_traj_subj['duration'].max()
    for g in gs:
        if g in obs_full_traj_subj['g_max_by_time'].values:
            obs_trans_time_g = obs_full_traj_subj.loc[obs_full_traj_subj['g_max_by_time'] == g, :]
            obs_trans_time_g.shape[0] >= 1
            obs_trans_time_g = obs_trans_time_g.iloc[0,:]
            rows.append(
                {
                    'g':g,
                    'event_observed':obs_trans_time_g['event_observed'],
                    't_observed_or_censored':obs_trans_time_g['duration'],
                    't_uncensored_alive':np.nan,
                    'missing_exact_t':False
                }
            )
        else:
            obs_trans_time_g_greater = obs_full_traj_subj.loc[
                (obs_full_traj_subj['g_max_by_time'] > g) & obs_full_traj_subj['event_observed'], :
            ]
            if obs_trans_time_g_greater.shape[0] >= 1:
                obs_trans_time_g_greater = obs_trans_time_g_greater.iloc[0,:]

                event_observed = True
                t_observed_or_censored = obs_trans_time_g_greater['duration']
                missing_exact_t = True

                obs_trans_time_g_smaller = obs_full_traj_subj.loc[
                    (obs_full_traj_subj['g_max_by_time'] < g) & obs_full_traj_subj['event_observed'], :
                ]
                if obs_trans_time_g_smaller.shape[0] >= 1:
                    obs_trans_time_g_smaller = obs_trans_time_g_smaller.iloc[-1,:]
                    t_uncensored_alive = obs_trans_time_g_smaller['duration']
                else:
                    t_uncensored_alive = 0
            else:
                event_observed = False
                t_observed_or_censored = t_max
                t_uncensored_alive = np.nan
                missing_exact_t = False
                
            rows.append(
                {
                    'g':g,
                    'event_observed':event_observed,
                    't_observed_or_censored':t_observed_or_censored,
                    't_uncensored_alive':t_uncensored_alive,
                    'missing_exact_t':missing_exact_t
                }
            )
    return pd.DataFrame(rows)
    
def get_all_grades_start_end_time(obs_full_traj, gs):
    assert all([g > 0 for g in gs])
    out = obs_full_traj.groupby('subject').apply(
        lambda df: _all_grades_start_end_time_subj(df, gs)
    ).reset_index(level=[0])
    return out



from sksurv.metrics import _check_estimate_2d
def _check_survival_all_subj(survival_test_all_subj):
    selector = survival_test_all_subj['missing_exact_t'].astype(bool)
    assert all(
        survival_test_all_subj.loc[selector, 't_uncensored_alive'] < 
        survival_test_all_subj.loc[selector, 't_observed_or_censored']
    )
    assert all(survival_test_all_subj.loc[selector, 'event_observed'])

def brier_score_at_g(subj_last_obs_time_train, survival_test_all_subj, estimate_all_subj, times, ipcw=True):
    # observed grades: 
    # if event_time <= t_grid: expect surv prob = 0 at t_grid
    # if event_time > t_grid: expect surv prob = 1 at t_grid

    # missing intermediate grades:
    # if event_time_next_grade <= t_grid: expect surv prob = 0 at t_grid
    # if event_time_prev_grade >= t_grid:  expect surv prob = 1 at t_grid

    # censored grades:
    # if event_time_c > t_grid: expect surv prob = 1 at t_grid

    _check_survival_all_subj(survival_test_all_subj)
    times = times.astype(float)
    test_event =  survival_test_all_subj['event_observed'].astype(bool)
    test_time_happened = survival_test_all_subj['t_observed_or_censored'].astype(float)
    test_time_alive = survival_test_all_subj ['t_uncensored_alive'].astype(float)
    missing_exact_t = survival_test_all_subj['missing_exact_t'].astype(bool)
    estimate, times = _check_estimate_2d(estimate_all_subj, test_time_happened, times, estimator="brier_score")
    if estimate.ndim == 1 and times.shape[0] == 1:
        estimate = estimate.reshape(-1, 1)
    if ipcw:
        # fit IPCW estimator
        subj_last_obs_time_train = subj_last_obs_time_train.sort_values(COLNAME_SURVIVAL_DURATION)
        subj_last_obs_time_train[COLNAME_SURVIVAL_EVENT_OBSERVED].iloc[0] = True # otherwise CensoringDistributionEstimator in integrated_brier_score won't run
        subj_last_obs_time_train[COLNAME_SURVIVAL_EVENT_OBSERVED] = subj_last_obs_time_train[COLNAME_SURVIVAL_EVENT_OBSERVED].astype(bool)
        y = subj_last_obs_time_train[[COLNAME_SURVIVAL_EVENT_OBSERVED,COLNAME_SURVIVAL_DURATION]].copy()
        y[COLNAME_SURVIVAL_DURATION] = y[COLNAME_SURVIVAL_DURATION].astype(float)
        y = y.to_records(index=False)
        cens = CensoringDistributionEstimator().fit(y)
        # calculate inverse probability of censoring weight at current time point t.
        prob_cens_t = cens.predict_proba(times)
        prob_cens_t[prob_cens_t == 0] = np.inf
        # calculate inverse probability of censoring weights at observed time point
        prob_cens_y = cens.predict_proba(test_time_happened)
        prob_cens_y[prob_cens_y == 0] = np.inf

    # Calculating the brier scores at each time point
    brier_scores = np.empty(times.shape[0], dtype=float)
    for i, t in enumerate(times):
        est = estimate[:, i]
        is_case = (test_time_happened <= t) & test_event 
        is_control_exact_t = (test_time_happened >= t) & ~test_event
        is_control_inexact_t = (test_time_alive >= t) & missing_exact_t
        is_control = is_control_exact_t | is_control_inexact_t
        N_certain = is_control.sum() + is_case.sum() 

        if ipcw:
            sum_cases = (np.square(est) * is_case.astype(int) / prob_cens_y).sum()
            sum_controls = (np.square(1.0 - est) * is_control.astype(int) / prob_cens_t[i]).sum()
            brier_scores[i] = (sum_cases + sum_controls)/N_certain
        else:
            sum_cases = np.square(est)[is_case].sum()
            sum_controls = np.square(1.0 - est)[is_control].sum()
            brier_scores[i] = (sum_cases + sum_controls)/N_certain

    return times, brier_scores


def get_integrated_brier_intrvl_imputed(obs_trans_time_all_g, out_model, df_last_obs_time_train, ipcw:Literal['by_grade', 'by_subj','without_ipcw'], max_grade=5, max_time=None):
    int_brier_all_grades = []
    out_model['true_prob']['g'] = out_model['true_prob']['g'].astype('float64').round(6)
    obs_trans_time_all_g['g'] = obs_trans_time_all_g['g'].astype('float64').round(6)
    df_last_obs_time_train['g'] = df_last_obs_time_train['g'].astype('float64').round(6)
    max_g_in_data = df_last_obs_time_train['g'].max()
    for g, obs_all_t_all_sbj in out_model['true_prob'].groupby('g'):
        if g > max_grade: continue
        survival_test_all_subj = obs_trans_time_all_g.loc[obs_trans_time_all_g['g'] == g,:]
        estimate_all_subj = obs_all_t_all_sbj.pivot(index='subj', columns='t', values='pred')
        assert set(estimate_all_subj.index) == set(survival_test_all_subj['subject'])
        
        estimate_all_subj = estimate_all_subj.loc[survival_test_all_subj['subject'].values,:]
        t_min = survival_test_all_subj['t_observed_or_censored'].min()
        t_max_test = survival_test_all_subj['t_observed_or_censored'].max()

        
        
        if ipcw == 'by_grade':
            df_last_obs_time_train_sub = df_last_obs_time_train.loc[
                df_last_obs_time_train['g'] == g, ['event_observed', 'duration']
            ].copy()
            ipcw_ = True
            if not max_time:
                t_max_train = df_last_obs_time_train.loc[df_last_obs_time_train['g'] == g, 'duration'].max()
                t_max = min(t_max_train, t_max_test)
            else:
                t_max = min(t_max_test, max_time)
        elif ipcw == 'by_subj':
            df_last_obs_time_train_sub = df_last_obs_time_train.loc[
                df_last_obs_time_train['g'] == max_g_in_data, ['event_observed', 'duration']
            ].copy()
            df_last_obs_time_train_sub['event_observed'] = False
            ipcw_ = True
            if not max_time:
                t_max_train = df_last_obs_time_train.loc[df_last_obs_time_train['g'] == max_g_in_data, 'duration'].max()
                t_max = min(t_max_train, t_max_test)
            else:
                t_max = min(t_max_test, max_time)
        elif ipcw == 'without_ipcw':
            df_last_obs_time_train_sub = df_last_obs_time_train
            ipcw_ = False
            if not max_time:
                t_max_train = df_last_obs_time_train.loc[df_last_obs_time_train['g'] == max_g_in_data, 'duration'].max()
                t_max = min(t_max_train, t_max_test)
            else:
                t_max = min(t_max_test, max_time)
        else:
            raise NotImplementedError
        times = estimate_all_subj.columns[(estimate_all_subj.columns >= t_min) & (estimate_all_subj.columns < t_max)]
        assert all(np.diff(times) > 0)
        estimate_all_subj=1-estimate_all_subj
        
        # try:
        t, brier = brier_score_at_g(
            df_last_obs_time_train_sub, 
            survival_test_all_subj=survival_test_all_subj, 
            estimate_all_subj=estimate_all_subj[times], 
            times=times, 
            ipcw=ipcw_
        )
        # except:
        #     print(g)
        #     raise ValueError
        int_brier = np.trapz(brier, t)/(times[-1] - times[0])
        int_brier_all_grades.append(int_brier)
    int_brier_all_grades = np.mean(int_brier_all_grades)
    return int_brier_all_grades


def mse_score_at_g(subj_last_obs_time_train, survival_test_all_subj, estimate_all_subj, truth_all_subj, times, ipcw=True):

    _check_survival_all_subj(survival_test_all_subj)
    times = times.astype(float)
    test_event =  survival_test_all_subj['event_observed'].astype(bool)
    test_time_happened = survival_test_all_subj['t_observed_or_censored'].astype(float)
    test_time_alive = survival_test_all_subj ['t_uncensored_alive'].astype(float)
    missing_exact_t = survival_test_all_subj['missing_exact_t'].astype(bool)
    estimate, times = _check_estimate_2d(estimate_all_subj, test_time_happened, times, estimator="brier_score")
    truth, times = _check_estimate_2d(truth_all_subj, test_time_happened, times, estimator="brier_score")
    if ipcw:
        # fit IPCW estimator
        subj_last_obs_time_train = subj_last_obs_time_train.sort_values(COLNAME_SURVIVAL_DURATION)
        subj_last_obs_time_train[COLNAME_SURVIVAL_EVENT_OBSERVED].iloc[0] = True # otherwise CensoringDistributionEstimator in integrated_brier_score won't run
        subj_last_obs_time_train[COLNAME_SURVIVAL_EVENT_OBSERVED] = subj_last_obs_time_train[COLNAME_SURVIVAL_EVENT_OBSERVED].astype(bool)
        y = subj_last_obs_time_train[[COLNAME_SURVIVAL_EVENT_OBSERVED,COLNAME_SURVIVAL_DURATION]].copy()
        y[COLNAME_SURVIVAL_DURATION] = y[COLNAME_SURVIVAL_DURATION].astype(float)
        y = y.to_records(index=False)
        cens = CensoringDistributionEstimator().fit(y)
        # calculate inverse probability of censoring weight at current time point t.
        prob_cens_t = cens.predict_proba(times)
        prob_cens_t[prob_cens_t == 0] = np.inf
        # calculate inverse probability of censoring weights at observed time point
        prob_cens_y = cens.predict_proba(test_time_happened)
        prob_cens_y[prob_cens_y == 0] = np.inf

    # Calculating the mse scores at each time point
    mse_scores = np.empty(times.shape[0], dtype=float)
    for i, t in enumerate(times):
        est = estimate[:, i]
        truth_t = truth[:, i]
        is_case = (test_time_happened <= t) & test_event 
        is_control_exact_t = (test_time_happened >= t) & ~test_event
        is_control_inexact_t = (test_time_alive >= t) & missing_exact_t
        is_control = is_control_exact_t | is_control_inexact_t
        is_certain_obs = is_control|is_case
        if ipcw:

            sum_cases_controls = np.mean(
                np.square(truth_t-est)* is_case.astype(int) / prob_cens_y + 
                np.square(truth_t-est)* is_control.astype(int) / prob_cens_t[i]
            ) 
            # what to do with interval censored data? the more interval-censored subjects, the smaller
            # the resulting score will be, this is a source of bias (artefact).
        else:
            sum_cases_controls = np.square(truth_t-est)[is_certain_obs].mean()
        mse_scores[i] = sum_cases_controls

    return times, mse_scores

def get_mse_vs_theory_certain_obs(obs_trans_time_all_g, out_model, df_last_obs_time_train, ipcw:Literal['by_grade', 'by_subj','without_ipcw'], max_grade=5, max_time=None):
    int_mse_all_grades = []
    max_g_in_data = df_last_obs_time_train['g'].max()
    for g, obs_all_t_all_sbj in out_model['true_prob'].groupby('g'):
        if g > max_grade: continue
        survival_test_all_subj = obs_trans_time_all_g.loc[obs_trans_time_all_g['g'] == g,:]
        estimate_all_subj = obs_all_t_all_sbj.pivot(index='subj', columns='t', values='pred')
        theory_all_subj = obs_all_t_all_sbj.pivot(index='subj', columns='t', values='truth')
        assert set(estimate_all_subj.index) == set(survival_test_all_subj['subject'])
        assert set(theory_all_subj.index) == set(survival_test_all_subj['subject'])
        
        estimate_all_subj = estimate_all_subj.loc[survival_test_all_subj['subject'].values,:]
        theory_all_subj = theory_all_subj.loc[survival_test_all_subj['subject'].values,:]
        t_min = survival_test_all_subj['t_observed_or_censored'].min()
        t_max_test = survival_test_all_subj['t_observed_or_censored'].max()

        if ipcw == 'by_grade':
            df_last_obs_time_train_sub = df_last_obs_time_train.loc[
                df_last_obs_time_train['g'] == g, ['event_observed', 'duration']
            ].copy()
            ipcw_ = True
        
            if not max_time:
                t_max_train = df_last_obs_time_train.loc[df_last_obs_time_train['g'] == g, 'duration'].max()
                t_max = min(t_max_train, t_max_test)
            else:
                t_max = min(t_max_test, max_time)
        elif ipcw == 'by_subj':
            df_last_obs_time_train_sub = df_last_obs_time_train.loc[
                df_last_obs_time_train['g'] == max_g_in_data, ['event_observed', 'duration']
            ].copy()
            df_last_obs_time_train_sub['event_observed'] = False
            ipcw_ = True
            if not max_time:
                t_max_train = df_last_obs_time_train.loc[df_last_obs_time_train['g'] == max_g_in_data, 'duration'].max()
                t_max = min(t_max_train, t_max_test)
            else:
                t_max = min(t_max_test, max_time)
        elif ipcw == 'without_ipcw':
            df_last_obs_time_train_sub = df_last_obs_time_train
            ipcw_ = False
            if not max_time:
                t_max_train = df_last_obs_time_train.loc[df_last_obs_time_train['g'] == max_g_in_data, 'duration'].max()
                t_max = min(t_max_train, t_max_test)
            else:
                t_max = min(t_max_test, max_time)
        else:
            raise NotImplementedError
        
        times = estimate_all_subj.columns[(estimate_all_subj.columns >= t_min) & (estimate_all_subj.columns < t_max)]
        assert all(np.diff(times) > 0)
        # try:
        t, mse = mse_score_at_g(
            subj_last_obs_time_train=df_last_obs_time_train_sub,
            survival_test_all_subj=survival_test_all_subj, 
            estimate_all_subj=estimate_all_subj[times], 
            truth_all_subj=theory_all_subj[times],
            times=times, 
            ipcw=ipcw_
        )
        # except:
        #     print(g)
        #     raise ValueError
        int_mse = np.trapz(mse, t)/(times[-1] - times[0])
        int_mse_all_grades.append(int_mse)
    int_mse_all_grades = np.mean(int_mse_all_grades)
    return int_mse_all_grades



def get_mse_vs_theory(out_model, t_max=None, g_max=None):

    df = out_model['true_prob']

    if t_max:
        df = df.loc[df['t'] <= t_max, :]
    if g_max:
        df = df.loc[df['g'] <= g_max, :]
    mse = np.square(df['truth'] - df['pred']).mean()
    return mse


def get_mae_vs_theory(out_model, t_max=None, g_max=None):

    df = out_model['true_prob']

    if t_max:
        df = df.loc[df['t'] <= t_max, :]
    if g_max:
        df = df.loc[df['g'] <= g_max, :]
    mse = np.abs(df['truth'] - df['pred']).mean()
    return mse


def get_kl_div_vs_theory(out_model, t_max=None, g_max=None):
    
    df_in_range = out_model['true_prob'].copy()

    if t_max:
        df_in_range = df_in_range.loc[df_in_range['t'] <= t_max, :]
    if g_max:
        df_in_range = df_in_range.loc[df_in_range['g'] <= g_max, :]
    kl_div_all_g_all_sub = []
    for (g, subj), obs_all_t_one_sbj in df_in_range.groupby(['g', 'subj']):
        df_asc_t = obs_all_t_one_sbj.sort_values('t')
        df_asc_t['pred'] = df_asc_t['pred'].diff()
        df_asc_t['truth'] = df_asc_t['truth'].diff()
        df_asc_t = df_asc_t.iloc[1::,:]
        thresh = 1e-6
        selector = df_asc_t['truth'] < thresh
        df_asc_t.loc[selector, 'truth'] = thresh
        selector = df_asc_t['pred'] < thresh
        df_asc_t.loc[selector, 'pred'] = thresh
        kl_div_ts = df_asc_t['pred']*np.log(df_asc_t['pred']/df_asc_t['truth'])
        kl_div_normed = np.trapz(x=df_asc_t['t'], y=kl_div_ts)/np.ptp(df_asc_t['t'])
        kl_div_all_g_all_sub.append(kl_div_normed)
    kl_div = np.mean(kl_div_all_g_all_sub)
    return kl_div


def get_ks_stats(out_model, t_max=None, g_max=None):
    
    df_in_range = out_model['true_prob'].copy()

    if t_max:
        df_in_range = df_in_range.loc[df_in_range['t'] <= t_max, :]
    if g_max:
        df_in_range = df_in_range.loc[df_in_range['g'] <= g_max, :]
    ks_all_g_all_sub = []
    for subj, obs_all_t_one_sbj in df_in_range.groupby(['subj']):
        ks = max((obs_all_t_one_sbj['pred'] - obs_all_t_one_sbj['truth']).abs())
        ks_all_g_all_sub.append(ks)
    ks_all_g_all_sub = np.mean(ks_all_g_all_sub)
    return ks_all_g_all_sub