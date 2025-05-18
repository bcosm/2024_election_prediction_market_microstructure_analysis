'''
performs granger causality tests
'''
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from tqdm import tqdm
import warnings

OUT_DIR = Path("./")
OUT_DIR.mkdir(exist_ok=True)

BIVARIATE_GRANGER_LAGS_TO_TEST = [1, 2, 3, 5, 8, 10]
MAX_LAGS_FOR_VAR_SELECTION = 12
ADF_SIGNIFICANCE_LEVEL = 0.05
GRANGER_SIGNIFICANCE_LEVEL = 0.05

MA_WINDOW_FOR_GAP = 5
MISPRICING_THRESHOLD_MA_DIFF = 0.015

EVENT_WINDOW_LENGTHS_BARS = [30, 60]
MIN_OBS_FOR_EVENT_GRANGER = max(BIVARIATE_GRANGER_LAGS_TO_TEST) + 15

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("Loading timebars.parquet...")
try:
    tb_df_all_markets = pd.read_parquet("timebars.parquet")
    tb_df_all_markets['time'] = pd.to_datetime(tb_df_all_markets['time'])
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading timebars.parquet: {e}")
    print("Please ensure 'timebars.parquet' exists and is correctly formatted with 'market' and 'time' columns.")
    exit()

def perform_adf_test(series, series_name=""):
    if series.empty or series.nunique(dropna=False) < 2:
        return 1.0
    try:
        cleaned_series = series.dropna()
        if cleaned_series.empty or cleaned_series.nunique() < 2:
            return 1.0
        result = adfuller(cleaned_series, autolag='AIC')
        p_value = result[1]
        return p_value
    except Exception as e:
        return 1.0

def make_series_stationary(series, series_name="", max_diffs=1, sig_level=ADF_SIGNIFICANCE_LEVEL):
    s_test = series.copy()
    is_stationary_final = False
    for d_count in range(max_diffs + 1):
        current_series_name_diff = f"{series_name} (d={d_count})"
        if d_count > 0:
            s_test = s_test.diff()
        
        p_value = perform_adf_test(s_test.dropna(), current_series_name_diff)
        
        if p_value <= sig_level:
            is_stationary_final = True
            return s_test.dropna(), d_count, is_stationary_final
            
    return s_test.dropna(), d_count, False


def select_var_optimal_lags(df_for_var, max_lags, criteria="aic", verbose=False):
    if df_for_var.shape[1] < 2:
        return 1
    if len(df_for_var) < max_lags + 3 * df_for_var.shape[1]:
        return 1
    try:
        var_model = VAR(df_for_var)
        selected_order = var_model.select_order(maxlags=max_lags, verbose=verbose)
        optimal_lag = getattr(selected_order, criteria)
        return optimal_lag if optimal_lag > 0 else 1
    except Exception as e:
        return 1

bivariate_granger_results_list = []
event_based_granger_results_list = []
multivariate_var_causality_results_list = []

if 'market' not in tb_df_all_markets.columns:
    print("Error: 'market' column missing in timebars.parquet.")
    exit()

unique_markets = tb_df_all_markets["market"].unique()

for market_name in tqdm(unique_markets, desc="Processing Markets", unit="market"):
    market_df_full = tb_df_all_markets[tb_df_all_markets["market"] == market_name]
    tqdm.write(f"\n===== Market: {market_name} =====")
    
    market_df = market_df_full.set_index('time').copy().sort_index()

    if len(market_df) < 200:
        tqdm.write(f"Skipping market {market_name} due to insufficient initial data points ({len(market_df)}).")
        continue

    market_df['deltaP'] = market_df['price'].diff()
    market_df['delta_RetailFlow'] = market_df['Retail'].diff()
    market_df['delta_InstFlow'] = market_df['Institutional'].diff()
    
    ma = market_df['price'].rolling(window=MA_WINDOW_FOR_GAP, min_periods=MA_WINDOW_FOR_GAP).mean()
    market_df['ma_diff'] = ((market_df['price'] - ma) / ma.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    market_df['delta_ma_diff'] = market_df['ma_diff'].diff()

    tqdm.write(f"[{market_name}] A. Running Bivariate Granger Tests (Multiple Lags)...")
    
    data_a1 = market_df[['deltaP', 'delta_RetailFlow']].dropna()
    if len(data_a1) > max(BIVARIATE_GRANGER_LAGS_TO_TEST) + 20:
        s_dp_a1, d_dp_a1, stat_dp_a1 = make_series_stationary(data_a1['deltaP'], f"{market_name}_deltaP_A1")
        s_rf_a1, d_rf_a1, stat_rf_a1 = make_series_stationary(data_a1['delta_RetailFlow'], f"{market_name}_deltaRetail_A1")
        df_a1_stat = pd.concat([s_dp_a1, s_rf_a1], axis=1).dropna()
        if not df_a1_stat.empty and len(df_a1_stat.columns)==2:
             df_a1_stat.columns = ['deltaP_stat', 'delta_RetailFlow_stat']
             if len(df_a1_stat) > max(BIVARIATE_GRANGER_LAGS_TO_TEST) + 10:
                try:
                    gc_res_a1 = grangercausalitytests(df_a1_stat[['deltaP_stat', 'delta_RetailFlow_stat']], BIVARIATE_GRANGER_LAGS_TO_TEST, verbose=False)
                    for lag in BIVARIATE_GRANGER_LAGS_TO_TEST:
                        if lag in gc_res_a1: bivariate_granger_results_list.append({'market': market_name, 'analysis_type': 'bivariate_full_series', 'test': 'deltaRetailFlow_to_deltaP', 'lag': lag, 'F_stat': gc_res_a1[lag][0]['ssr_ftest'][0], 'p_value': gc_res_a1[lag][0]['ssr_ftest'][1], 'stat_var1':stat_rf_a1, 'diffs_var1':d_rf_a1, 'stat_var2':stat_dp_a1, 'diffs_var2':d_dp_a1})
                    gc_res_a1_rev = grangercausalitytests(df_a1_stat[['delta_RetailFlow_stat', 'deltaP_stat']], BIVARIATE_GRANGER_LAGS_TO_TEST, verbose=False)
                    for lag in BIVARIATE_GRANGER_LAGS_TO_TEST:
                        if lag in gc_res_a1_rev: bivariate_granger_results_list.append({'market': market_name, 'analysis_type': 'bivariate_full_series', 'test': 'deltaP_to_deltaRetailFlow', 'lag': lag, 'F_stat': gc_res_a1_rev[lag][0]['ssr_ftest'][0], 'p_value': gc_res_a1_rev[lag][0]['ssr_ftest'][1], 'stat_var1':stat_dp_a1, 'diffs_var1':d_dp_a1, 'stat_var2':stat_rf_a1, 'diffs_var2':d_rf_a1})
                except Exception as e: tqdm.write(f"  Error in Bivariate Granger A1 for {market_name}: {e}")
    else: tqdm.write(f"  Skipping delta_RetailFlow vs deltaP for {market_name}, not enough data.")

    data_a2 = market_df[['delta_ma_diff', 'delta_InstFlow']].dropna()
    if len(data_a2) > max(BIVARIATE_GRANGER_LAGS_TO_TEST) + 20:
        s_dmd_a2, d_dmd_a2, stat_dmd_a2 = make_series_stationary(data_a2['delta_ma_diff'], f"{market_name}_deltaMaDiff_A2")
        s_if_a2, d_if_a2, stat_if_a2 = make_series_stationary(data_a2['delta_InstFlow'], f"{market_name}_deltaInstFlow_A2")
        df_a2_stat = pd.concat([s_dmd_a2, s_if_a2], axis=1).dropna()
        if not df_a2_stat.empty and len(df_a2_stat.columns)==2:
            df_a2_stat.columns = ['delta_ma_diff_stat', 'delta_InstFlow_stat']
            if len(df_a2_stat) > max(BIVARIATE_GRANGER_LAGS_TO_TEST) + 10:
                try:
                    gc_res_a2 = grangercausalitytests(df_a2_stat[['delta_ma_diff_stat', 'delta_InstFlow_stat']], BIVARIATE_GRANGER_LAGS_TO_TEST, verbose=False)
                    for lag in BIVARIATE_GRANGER_LAGS_TO_TEST:
                        if lag in gc_res_a2: bivariate_granger_results_list.append({'market': market_name, 'analysis_type': 'bivariate_full_series', 'test': 'deltaInstFlow_to_deltaMaDiff', 'lag': lag, 'F_stat': gc_res_a2[lag][0]['ssr_ftest'][0], 'p_value': gc_res_a2[lag][0]['ssr_ftest'][1], 'stat_var1':stat_if_a2, 'diffs_var1':d_if_a2, 'stat_var2':stat_dmd_a2, 'diffs_var2':d_dmd_a2})
                    gc_res_a2_rev = grangercausalitytests(df_a2_stat[['delta_InstFlow_stat', 'delta_ma_diff_stat']], BIVARIATE_GRANGER_LAGS_TO_TEST, verbose=False)
                    for lag in BIVARIATE_GRANGER_LAGS_TO_TEST:
                        if lag in gc_res_a2_rev: bivariate_granger_results_list.append({'market': market_name, 'analysis_type': 'bivariate_full_series', 'test': 'deltaMaDiff_to_deltaInstFlow', 'lag': lag, 'F_stat': gc_res_a2_rev[lag][0]['ssr_ftest'][0], 'p_value': gc_res_a2_rev[lag][0]['ssr_ftest'][1], 'stat_var1':stat_dmd_a2, 'diffs_var1':d_dmd_a2, 'stat_var2':stat_if_a2, 'diffs_var2':d_if_a2})
                except Exception as e: tqdm.write(f"  Error in Bivariate Granger A2 for {market_name}: {e}")
    else: tqdm.write(f"  Skipping delta_InstFlow vs delta_ma_diff for {market_name}, not enough data.")


    tqdm.write(f"[{market_name}] B. Running Event-Based Granger Tests...")
    mispricing_event_indices = market_df.index[np.abs(market_df['ma_diff']) > MISPRICING_THRESHOLD_MA_DIFF]
    
    if not mispricing_event_indices.empty:
        for window_len_bars in EVENT_WINDOW_LENGTHS_BARS:
            tqdm.write(f"  [{market_name}] Event-based analysis for window length: {window_len_bars} bars")
            for event_idx_loc in tqdm(range(len(mispricing_event_indices)), desc=f"Events(W={window_len_bars})", leave=False):
                event_time = mispricing_event_indices[event_idx_loc]
                try:
                    event_iloc = market_df.index.get_loc(event_time)
                except KeyError:
                    continue

                if event_iloc + 1 + window_len_bars > len(market_df):
                    continue 
                
                segment_start_iloc = event_iloc + 1
                segment_end_iloc = event_iloc + 1 + window_len_bars
                
                event_segment = market_df.iloc[segment_start_iloc:segment_end_iloc][['delta_InstFlow', 'delta_ma_diff']].copy()
                event_segment = event_segment.dropna()

                if len(event_segment) < MIN_OBS_FOR_EVENT_GRANGER:
                    continue

                s_dmd_ev, d_dmd_ev, stat_dmd_ev = make_series_stationary(event_segment['delta_ma_diff'], f"{market_name}_EvDeltaMaDiff")
                s_if_ev, d_if_ev, stat_if_ev = make_series_stationary(event_segment['delta_InstFlow'], f"{market_name}_EvDeltaInstFlow")
                
                df_ev_stat = pd.concat([s_dmd_ev, s_if_ev], axis=1).dropna()
                if not df_ev_stat.empty and len(df_ev_stat.columns)==2:
                    df_ev_stat.columns = ['delta_ma_diff_stat', 'delta_InstFlow_stat']

                    if len(df_ev_stat) > max(BIVARIATE_GRANGER_LAGS_TO_TEST) + 10 :
                        lags_to_test_event = [l for l in BIVARIATE_GRANGER_LAGS_TO_TEST if l < len(df_ev_stat) / 3 and l < MIN_OBS_FOR_EVENT_GRANGER - 5]
                        if not lags_to_test_event: lags_to_test_event = [1]
                        lags_to_test_event = sorted(list(set(lags_to_test_event)))
                        if not lags_to_test_event: continue


                        try:
                            gc_res_ev = grangercausalitytests(df_ev_stat[['delta_ma_diff_stat', 'delta_InstFlow_stat']], lags_to_test_event, verbose=False)
                            for lag in lags_to_test_event:
                                if lag in gc_res_ev:
                                    event_based_granger_results_list.append({
                                        'market': market_name, 'event_time': event_time, 'window_length': window_len_bars,
                                        'test': 'deltaInstFlow_to_deltaMaDiff_EVENT', 'lag': lag,
                                        'F_stat': gc_res_ev[lag][0]['ssr_ftest'][0], 'p_value': gc_res_ev[lag][0]['ssr_ftest'][1],
                                        'stat_var1':stat_if_ev, 'diffs_var1':d_if_ev, 'stat_var2':stat_dmd_ev, 'diffs_var2':d_dmd_ev,
                                        'n_obs_segment': len(df_ev_stat)
                                    })
                        except Exception: pass
    else:
        tqdm.write(f"  [{market_name}] No mispricing events found for event-based analysis with threshold {MISPRICING_THRESHOLD_MA_DIFF}.")


    tqdm.write(f"[{market_name}] C. Running Multivariate VAR Granger Causality Tests...")
    
    data_c1_raw = market_df[['deltaP', 'delta_RetailFlow', 'delta_InstFlow']].dropna()
    if len(data_c1_raw) > MAX_LAGS_FOR_VAR_SELECTION + 30:
        s_dp_c1, d_dp_c1, stat_dp_c1 = make_series_stationary(data_c1_raw['deltaP'], f"{market_name}_VAR_deltaP")
        s_rf_c1, d_rf_c1, stat_rf_c1 = make_series_stationary(data_c1_raw['delta_RetailFlow'], f"{market_name}_VAR_deltaRetailFlow")
        s_if_c1, d_if_c1, stat_if_c1 = make_series_stationary(data_c1_raw['delta_InstFlow'], f"{market_name}_VAR_deltaInstFlow")
        
        df_c1_stat = pd.concat([s_dp_c1, s_rf_c1, s_if_c1], axis=1).dropna()
        if not df_c1_stat.empty and len(df_c1_stat.columns)==3:
            df_c1_stat.columns = ['deltaP_stat', 'delta_RetailFlow_stat', 'delta_InstFlow_stat']

            if len(df_c1_stat) > MAX_LAGS_FOR_VAR_SELECTION + 20:
                try:
                    var_lags_c1 = select_var_optimal_lags(df_c1_stat, MAX_LAGS_FOR_VAR_SELECTION)
                    var_model_c1_fit = VAR(df_c1_stat).fit(maxlags=var_lags_c1, verbose=False)
                    
                    tests_to_run_c1 = [
                        ('deltaP_stat', ['delta_RetailFlow_stat'], 'RetailFlow_on_DeltaP_VAR'),
                        ('deltaP_stat', ['delta_InstFlow_stat'], 'InstFlow_on_DeltaP_VAR'),
                        ('deltaP_stat', ['delta_RetailFlow_stat', 'delta_InstFlow_stat'], 'BothFlows_on_DeltaP_VAR')
                    ]
                    for dep_var, causing_vars, test_name in tests_to_run_c1:
                        res_test = var_model_c1_fit.test_causality(dep_var, causing_vars, kind='f', signif=GRANGER_SIGNIFICANCE_LEVEL)
                        multivariate_var_causality_results_list.append({
                            'market':market_name, 'model_vars': 'deltaP~dRF+dIF', 'dependent_var':dep_var, 
                            'causing_vars':",".join(causing_vars), 'test_name': test_name, 'lags':var_lags_c1, 
                            'F_stat':res_test.test_statistic, 'p_value':res_test.pvalue, 'df_num': res_test.df_num, 'df_denom':res_test.df_denom
                        })
                except Exception as e: tqdm.write(f"  Error in VAR C1 for {market_name}: {e}")
    else: tqdm.write(f"  Skipping VAR Price Dynamics for {market_name}, not enough data.")


    data_c2_raw = market_df[['delta_ma_diff', 'delta_InstFlow', 'delta_RetailFlow']].dropna()
    if len(data_c2_raw) > MAX_LAGS_FOR_VAR_SELECTION + 30:
        s_dmd_c2, d_dmd_c2, stat_dmd_c2 = make_series_stationary(data_c2_raw['delta_ma_diff'], f"{market_name}_VAR_deltaMaDiff")
        s_if_c2, d_if_c2, stat_if_c2 = make_series_stationary(data_c2_raw['delta_InstFlow'], f"{market_name}_VAR_deltaInstFlow_C2")
        s_rf_c2, d_rf_c2, stat_rf_c2 = make_series_stationary(data_c2_raw['delta_RetailFlow'], f"{market_name}_VAR_deltaRetailFlow_C2")
        
        df_c2_stat = pd.concat([s_dmd_c2, s_if_c2, s_rf_c2], axis=1).dropna()
        if not df_c2_stat.empty and len(df_c2_stat.columns)==3:
            df_c2_stat.columns = ['delta_ma_diff_stat', 'delta_InstFlow_stat', 'delta_RetailFlow_stat']

            if len(df_c2_stat) > MAX_LAGS_FOR_VAR_SELECTION + 20:
                try:
                    var_lags_c2 = select_var_optimal_lags(df_c2_stat, MAX_LAGS_FOR_VAR_SELECTION)
                    var_model_c2_fit = VAR(df_c2_stat).fit(maxlags=var_lags_c2, verbose=False)

                    tests_to_run_c2 = [
                        ('delta_ma_diff_stat', ['delta_InstFlow_stat'], 'InstFlow_on_DeltaMaDiff_VAR'),
                        ('delta_ma_diff_stat', ['delta_RetailFlow_stat'], 'RetailFlow_on_DeltaMaDiff_VAR'),
                        ('delta_ma_diff_stat', ['delta_InstFlow_stat', 'delta_RetailFlow_stat'], 'BothFlows_on_DeltaMaDiff_VAR')
                    ]
                    for dep_var, causing_vars, test_name in tests_to_run_c2:
                        res_test = var_model_c2_fit.test_causality(dep_var, causing_vars, kind='f', signif=GRANGER_SIGNIFICANCE_LEVEL)
                        multivariate_var_causality_results_list.append({
                            'market':market_name, 'model_vars': 'dGap~dIF+dRF', 'dependent_var':dep_var, 
                            'causing_vars':",".join(causing_vars), 'test_name': test_name, 'lags':var_lags_c2, 
                            'F_stat':res_test.test_statistic, 'p_value':res_test.pvalue, 'df_num': res_test.df_num, 'df_denom':res_test.df_denom
                        })
                except Exception as e: tqdm.write(f"  Error in VAR C2 for {market_name}: {e}")
    else: tqdm.write(f"  Skipping VAR Gap Dynamics for {market_name}, not enough data.")


if bivariate_granger_results_list:
    pd.DataFrame(bivariate_granger_results_list).sort_values(['market', 'test', 'lag']).to_csv(OUT_DIR / "bivariate_granger_summary_multiple_lags.csv", index=False)
    tqdm.write(f"\nBivariate Granger summary saved to {OUT_DIR / 'bivariate_granger_summary_multiple_lags.csv'}")

if event_based_granger_results_list:
    pd.DataFrame(event_based_granger_results_list).sort_values(['market', 'event_time', 'window_length', 'test', 'lag']).to_csv(OUT_DIR / "event_based_granger_summary.csv", index=False)
    tqdm.write(f"Event-based Granger summary saved to {OUT_DIR / 'event_based_granger_summary.csv'}")

if multivariate_var_causality_results_list:
    pd.DataFrame(multivariate_var_causality_results_list).sort_values(['market', 'model_vars', 'dependent_var', 'causing_vars']).to_csv(OUT_DIR / "multivariate_var_causality_summary.csv", index=False)
    tqdm.write(f"Multivariate VAR causality summary saved to {OUT_DIR / 'multivariate_var_causality_summary.csv'}")

if not bivariate_granger_results_list and not event_based_granger_results_list and not multivariate_var_causality_results_list:
    tqdm.write("\nNo Granger causality tests were successfully completed across all enhancements.")

