"""analyzes market time series data using bivariate var models and an event study of flow accumulation following mispricing events"""
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

OUT_DIR = Path("./")
OUT_DIR.mkdir(exist_ok=True)

MAX_LAGS_FOR_VAR_INITIAL_GUESS = 10
MIN_LAGS_FOR_VAR_SEARCH = 1
MAX_LAGS_FOR_VALID_VAR_SEARCH = 12
RESIDUAL_WHITENESS_TEST_NLAGS = 20
ADF_SIGNIFICANCE_LEVEL = 0.05
GRANGER_SIGNIFICANCE_LEVEL = 0.05
MODEL_VALIDATION_SIGNIFICANCE = 0.05

MA_WINDOW_FOR_GAP = 5
MISPRICING_THRESHOLD_MA_DIFF = 0.015

EVENT_WINDOW_LENGTHS_BARS_FLOW_STUDY = [30, 60]
SUCCESSFUL_CORRECTION_THRESHOLD_PCT = 0.75

IRF_PERIODS = 20

USE_DIFFERENCED_FLOW_FOR_VAR = True
USE_DIFFERENCED_GAP_FOR_VAR = True

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.ValueWarning)

print("Loading timebars.parquet...")
try:
    tb_df_all_markets = pd.read_parquet("timebars.parquet")
    tb_df_all_markets['time'] = pd.to_datetime(tb_df_all_markets['time'])
except Exception as e:
    print(f"Error loading timebars.parquet: {e}")
    exit()

def perform_adf_test_detailed(series, series_name=""):
    if series.empty or series.nunique(dropna=False) < 2: return 1.0, f"{series_name}: Series empty or constant."
    try:
        cleaned_series = series.dropna();
        if cleaned_series.empty or cleaned_series.nunique() < 2: return 1.0, f"{series_name}: Series empty or constant after dropna."
        result = adfuller(cleaned_series, regression='c', autolag='AIC'); p_value = result[1]
        return p_value, f"ADF for {series_name}: p={p_value:.4f}, n_obs={len(cleaned_series)}"
    except Exception as e: return 1.0, f"ADF test failed for {series_name}: {e}"

def make_series_stationary_for_var(input_series, series_name="", max_diffs=1, sig_level=ADF_SIGNIFICANCE_LEVEL):
    current_series = input_series.copy(); adf_log_full = []; is_stationary = False; final_diff_count = 0
    for d_count in range(max_diffs + 1):
        series_to_test = current_series.dropna()
        p_value, adf_msg = perform_adf_test_detailed(series_to_test, f"{series_name}_d{d_count}")
        adf_log_full.append(adf_msg)
        if p_value <= sig_level:
            is_stationary = True; final_diff_count = d_count; break
        if d_count < max_diffs: current_series = current_series.diff()
        else: final_diff_count = d_count
    return current_series, final_diff_count, is_stationary, adf_log_full

def get_initial_var_lags(df_for_var, max_lags, criteria="aic", verbose=False):
    if df_for_var.shape[1] < 2 or len(df_for_var) < max_lags + 10 * df_for_var.shape[1]: return MIN_LAGS_FOR_VAR_SEARCH
    try:
        var_model_select = VAR(df_for_var); selected_order = var_model_select.select_order(maxlags=max_lags, verbose=verbose)
        optimal_lag = getattr(selected_order, criteria)
        return optimal_lag if optimal_lag > 0 else MIN_LAGS_FOR_VAR_SEARCH
    except Exception: return MIN_LAGS_FOR_VAR_SEARCH

stationarity_logs_list = []
bivar_var_fitting_log_list = []
bivar_var_valid_diagnostics_list = []
bivar_var_causality_results_list = []
event_flow_dynamics_list = []

if 'market' not in tb_df_all_markets.columns: print("Error: 'market' column missing."); exit()
unique_markets = tb_df_all_markets["market"].unique()

for market_name in tqdm(unique_markets, desc="Total Market Progress", unit="market", position=0):
    market_df_full = tb_df_all_markets[tb_df_all_markets["market"] == market_name]
    tqdm.write(f"\n===== Market: {market_name} (Option 2 Deep Dive) =====")
    
    df = market_df_full.set_index('time').copy().sort_index()
    if len(df) < 300: tqdm.write(f"Skipping {market_name}: insufficient data ({len(df)})."); continue

    df['deltaP_orig'] = df['price'].diff()
    df['RetailFlow_orig'] = df['Retail']
    df['InstFlow_orig'] = df['Institutional']
    ma = df['price'].rolling(window=MA_WINDOW_FOR_GAP, min_periods=MA_WINDOW_FOR_GAP).mean()
    df['ma_diff_orig'] = ((df['price'] - ma) / ma.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    
    bivar_vars_defs = {
        'deltaP': df['deltaP_orig'],
        'delta_RetailFlow': df['RetailFlow_orig'].diff() if USE_DIFFERENCED_FLOW_FOR_VAR else df['RetailFlow_orig'],
        'delta_InstFlow': df['InstFlow_orig'].diff() if USE_DIFFERENCED_FLOW_FOR_VAR else df['InstFlow_orig'],
        'Gap_var': df['ma_diff_orig'].diff() if USE_DIFFERENCED_GAP_FOR_VAR else df['ma_diff_orig']
    }
    gap_var_name_for_bivar = 'delta_ma_diff' if USE_DIFFERENCED_GAP_FOR_VAR else 'ma_diff'

    bivariate_var_pairs_config = [
        {'name': f'deltaP_vs_deltaRetailFlow', 'vars': ['deltaP', 'delta_RetailFlow']},
        {'name': f'deltaP_vs_deltaInstFlow', 'vars': ['deltaP', 'delta_InstFlow']},
        {'name': f'{gap_var_name_for_bivar}_vs_deltaInstFlow', 'vars': ['Gap_var', 'delta_InstFlow']},
        {'name': f'{gap_var_name_for_bivar}_vs_deltaRetailFlow', 'vars': ['Gap_var', 'delta_RetailFlow']},
    ]

    for pair_config in tqdm(bivariate_var_pairs_config, desc=f"Bivariate Pairs for {market_name}", unit="pair", leave=False, position=1):
        current_pair_data_raw = {key: bivar_vars_defs[key] for key in pair_config['vars']}
        
        current_pair_stat_series = {}
        all_series_proc_completed = True
        for var_key, series_data in current_pair_data_raw.items():
            s_series, d_count, is_stat, adf_log = make_series_stationary_for_var(series_data, f"{market_name}_{var_key}_{pair_config['name']}")
            current_pair_stat_series[var_key] = s_series
            stationarity_logs_list.extend([{'market':market_name, 'analysis_type':'bivariate_VAR', 'variable_key':var_key, 'series_name_in_model':pair_config['name'], 'diffs':d_count, 'is_stationary':is_stat, 'log':l} for l in adf_log])
            if not is_stat: all_series_proc_completed = False
        
        if not all_series_proc_completed:
            continue
            
        df_for_bivar = pd.concat(current_pair_stat_series.values(), axis=1, keys=current_pair_stat_series.keys()).dropna()
        df_for_bivar.columns = [f"{col}_stat" for col in df_for_bivar.columns]

        if len(df_for_bivar) < MAX_LAGS_FOR_VALID_VAR_SEARCH + 50:
            continue

        valid_bivar_model_found = False; best_bivar_model_fit = None; actual_bivar_lags_used = 0
        initial_bivar_lags_aic = get_initial_var_lags(df_for_bivar, MAX_LAGS_FOR_VAR_INITIAL_GUESS)
        
        for p_lags in tqdm(range(max(MIN_LAGS_FOR_VAR_SEARCH, initial_bivar_lags_aic), MAX_LAGS_FOR_VALID_VAR_SEARCH + 1), desc=f"Lags for {pair_config['name']}", leave=False, position=2):
            log_entry = {'market': market_name, 'model_name': pair_config['name'], 'lags_tried': p_lags}
            if len(df_for_bivar) < p_lags + 15:
                 log_entry.update({'error': 'Insufficient observations for this lag order.'}); bivar_var_fitting_log_list.append(log_entry); continue
            try:
                bivar_fit_iter = VAR(df_for_bivar).fit(maxlags=p_lags, ic=None, verbose=False)
                is_stable_iter = bivar_fit_iter.is_stable(verbose=False)
                
                nlags_whiteness = min(RESIDUAL_WHITENESS_TEST_NLAGS, len(df_for_bivar)//2 -1 )
                if nlags_whiteness <= p_lags : nlags_whiteness = p_lags + 5

                whiteness_pvalue = np.nan
                if len(df_for_bivar) > nlags_whiteness + p_lags * df_for_bivar.shape[1]:
                    resid_whiteness_iter = bivar_fit_iter.test_whiteness(nlags=nlags_whiteness, signif=MODEL_VALIDATION_SIGNIFICANCE, adjusted=True)
                    whiteness_pvalue = resid_whiteness_iter.pvalue if hasattr(resid_whiteness_iter, 'pvalue') else float(str(resid_whiteness_iter.summary()).split("Prob(RLM)")[1].split(":")[1].split("\n")[0].strip())

                log_entry.update({'is_stable': is_stable_iter, 'resid_whiteness_pvalue': whiteness_pvalue, 'aic': bivar_fit_iter.aic})
                bivar_var_fitting_log_list.append(log_entry)

                if is_stable_iter and (pd.isna(whiteness_pvalue) or whiteness_pvalue > MODEL_VALIDATION_SIGNIFICANCE):
                    valid_bivar_model_found = True; best_bivar_model_fit = bivar_fit_iter; actual_bivar_lags_used = p_lags; break
            except Exception as e:
                log_entry.update({'error': str(e)}); bivar_var_fitting_log_list.append(log_entry)
                if "positive definite" in str(e).lower() or "degrees of freedom" in str(e).lower(): break
        
        if valid_bivar_model_found and best_bivar_model_fit:
            diag_entry = {'market': market_name, 'model_name': pair_config['name'], 'lags_used': actual_bivar_lags_used,
                          'is_stable': best_bivar_model_fit.is_stable(verbose=False), 'aic': best_bivar_model_fit.aic,
                          'bic': best_bivar_model_fit.bic, 'resid_whiteness_pvalue': whiteness_pvalue}
            for i_eq, eq_name in enumerate(df_for_bivar.columns): diag_entry[f'dw_resid_{eq_name}'] = durbin_watson(best_bivar_model_fit.resid.iloc[:,i_eq])
            bivar_var_valid_diagnostics_list.append(diag_entry)

            var1_name, var2_name = df_for_bivar.columns[0], df_for_bivar.columns[1]
            try:
                gc_res1 = best_bivar_model_fit.test_causality(var1_name, [var2_name], kind='f')
                bivar_var_causality_results_list.append({'market':market_name, 'model_name': pair_config['name'], 'dependent_var':var1_name, 'causing_var':var2_name, 'lags':actual_bivar_lags_used, 'F_stat':gc_res1.test_statistic, 'p_value':gc_res1.pvalue})
                gc_res2 = best_bivar_model_fit.test_causality(var2_name, [var1_name], kind='f')
                bivar_var_causality_results_list.append({'market':market_name, 'model_name': pair_config['name'], 'dependent_var':var2_name, 'causing_var':var1_name, 'lags':actual_bivar_lags_used, 'F_stat':gc_res2.test_statistic, 'p_value':gc_res2.pvalue})
            except Exception as e: pass

            try:
                irf = best_bivar_model_fit.irf(periods=IRF_PERIODS)
                irf_plot = irf.plot(orth=False)
                irf_plot.suptitle(f'Impulse Responses: {market_name} - {pair_config["name"]} (VAR({actual_bivar_lags_used}))', fontsize=10)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                irf_fig_path = OUT_DIR / f"IRF_{market_name}_{pair_config['name']}_L{actual_bivar_lags_used}.png"
                irf_plot.savefig(irf_fig_path, dpi=150)
                plt.close('all')
            except Exception as e: pass
        else:
            pass

    df_event_study = df[['RetailFlow_orig', 'InstFlow_orig', 'ma_diff_orig', 'price']].copy()
    df_event_study.rename(columns={'RetailFlow_orig':'RetailFlow', 'InstFlow_orig':'InstFlow'}, inplace=True)
    
    mispricing_event_times = df_event_study.index[np.abs(df_event_study['ma_diff_orig']) > MISPRICING_THRESHOLD_MA_DIFF]

    if not mispricing_event_times.empty:
        for event_time_start in tqdm(mispricing_event_times, desc=f"Mispricing Events for {market_name}", unit="event", leave=False, position=1):
            initial_ma_diff = df_event_study.loc[event_time_start, 'ma_diff_orig']
            if pd.isna(initial_ma_diff): continue

            for window_len in EVENT_WINDOW_LENGTHS_BARS_FLOW_STUDY:
                window_end_time = event_time_start + pd.Timedelta(minutes=window_len)
                event_window_df = df_event_study.loc[event_time_start : window_end_time].copy()
                
                if event_window_df.empty or len(event_window_df) < 2: continue

                actual_window_start_time = event_window_df.index[0]
                actual_window_end_time = event_window_df.index[-1]
                
                cum_retail_flow = event_window_df['RetailFlow'].iloc[1:].sum()
                cum_inst_flow = event_window_df['InstFlow'].iloc[1:].sum()
                
                final_ma_diff = event_window_df['ma_diff_orig'].iloc[-1]
                
                gap_closed_pct = np.nan
                if abs(initial_ma_diff) > 1e-9:
                    gap_closed_pct = 1 - (abs(final_ma_diff) / abs(initial_ma_diff))
                
                is_successful = gap_closed_pct >= SUCCESSFUL_CORRECTION_THRESHOLD_PCT if not pd.isna(gap_closed_pct) else False
                
                event_flow_dynamics_list.append({
                    'market': market_name,
                    'event_time_start': actual_window_start_time,
                    'window_length_cfg': window_len,
                    'actual_window_duration_min': (actual_window_end_time - actual_window_start_time).total_seconds() / 60,
                    'initial_ma_diff': initial_ma_diff,
                    'final_ma_diff': final_ma_diff,
                    'gap_closed_pct': gap_closed_pct,
                    'is_successful_correction': is_successful,
                    'cumulative_retail_flow_in_window': cum_retail_flow,
                    'cumulative_inst_flow_in_window': cum_inst_flow,
                    'n_bars_in_window': len(event_window_df)-1
                })
    else:
        tqdm.write(f"    No mispricing events found for {market_name} with threshold {MISPRICING_THRESHOLD_MA_DIFF}.")

pd.DataFrame(stationarity_logs_list).sort_values(['market', 'analysis_type', 'variable_key']).to_csv(
    OUT_DIR / "bivar_var_stationarity_logs.csv", index=False)
pd.DataFrame(bivar_var_fitting_log_list).sort_values(['market', 'model_name', 'lags_tried']).to_csv(
    OUT_DIR / "bivar_var_fitting_log.csv", index=False)
if bivar_var_valid_diagnostics_list:
    pd.DataFrame(bivar_var_valid_diagnostics_list).sort_values(['market', 'model_name']).to_csv(
        OUT_DIR / "bivar_var_VALID_model_diagnostics.csv", index=False)
if bivar_var_causality_results_list:
    pd.DataFrame(bivar_var_causality_results_list).sort_values(['market', 'model_name', 'dependent_var', 'causing_var']).to_csv(
        OUT_DIR / "bivar_var_VALID_causality_summary.csv", index=False)

if event_flow_dynamics_list:
    df_event_flow_summary = pd.DataFrame(event_flow_dynamics_list)
    df_event_flow_summary.sort_values(['market', 'event_time_start', 'window_length_cfg']).to_csv(
        OUT_DIR / "event_flow_dynamics_summary.csv", index=False)

    for market in df_event_flow_summary['market'].unique():
        for window_len_cfg in df_event_flow_summary['window_length_cfg'].unique():
            subset = df_event_flow_summary[(df_event_flow_summary['market'] == market) & (df_event_flow_summary['window_length_cfg'] == window_len_cfg)]
            if len(subset) < 10: continue

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.boxplot(x='is_successful_correction', y='cumulative_retail_flow_in_window', data=subset, palette=['#FF9999', '#99FF99'])
            plt.title(f'Retail Flow (Window={window_len_cfg}m)\n{market}')
            plt.ylabel('Cumulative Net Retail Flow')
            plt.xlabel('Correction Successful?')
            
            plt.subplot(1, 2, 2)
            sns.boxplot(x='is_successful_correction', y='cumulative_inst_flow_in_window', data=subset, palette=['#FF9999', '#99FF99'])
            plt.title(f'Institutional Flow (Window={window_len_cfg}m)\n{market}')
            plt.ylabel('Cumulative Net Institutional Flow')
            plt.xlabel('Correction Successful?')
            
            plt.suptitle(f"Cumulative Flow by Correction Success - {market}, {window_len_cfg} min window", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plot_path = OUT_DIR / f"AvgCumFlow_{market}_W{window_len_cfg}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close('all')
else:
    tqdm.write("\nNo event flow dynamics data generated.")

