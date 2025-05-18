"""performs z-score based analysis"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from tqdm import tqdm
import warnings
import datetime

TIMEBARS_FILE_PATH = Path("timebars.parquet")

ZSCORE_MAIN_OUTPUT_DIR = Path("./")
ZSCORE_DETAILED_SEGMENT_REPORTS_DIR = ZSCORE_MAIN_OUTPUT_DIR / "detailed_segment_reports_ZSCORE"
ZSCORE_GRANGER_OUTPUT_DIR = ZSCORE_MAIN_OUTPUT_DIR / "granger_causality_ZSCORE"


ZSCORE_MA_WINDOW = 5
ZSCORE_STD_ROLLING_WINDOW = 60
ZSCORE_THRESHOLD = 2.0

ZSCORE_EVENT_FLOW_WINDOWS = [30, 60]
ZSCORE_SUCCESS_THRESHOLD_PCT = 0.75

GRANGER_LAGS_TO_TEST = [1, 2, 3, 5, 8, 10]
GRANGER_ADF_SIGNIFICANCE_LEVEL = 0.05
GRANGER_EVENT_MIN_OBS_FOR_TEST = max(GRANGER_LAGS_TO_TEST) + 15


ZSCORE_MAIN_OUTPUT_DIR.mkdir(exist_ok=True)
ZSCORE_DETAILED_SEGMENT_REPORTS_DIR.mkdir(exist_ok=True)
ZSCORE_GRANGER_OUTPUT_DIR.mkdir(exist_ok=True)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def calculate_zscore_ma_diff(df_market, ma_window_for_pct, std_rolling_window):
    df = df_market.copy()
    
    if len(df['price'].dropna()) < ma_window_for_pct:
        print(f"Warning: Not enough data for MA {ma_window_for_pct} in Z-score calc for market {df['market'].iloc[0] if not df.empty else 'Unknown'}. Assigning NaNs.")
        df['ma_diff_pct'] = np.nan
    else:
        ma = df['price'].rolling(window=ma_window_for_pct, min_periods=max(1, ma_window_for_pct // 2)).mean()
        df['ma_diff_pct'] = (df['price'] - ma) / ma.replace(0, np.nan)
        df['ma_diff_pct'].replace([np.inf, -np.inf], np.nan, inplace=True)

    if df['ma_diff_pct'].isna().all() or len(df['ma_diff_pct'].dropna()) < std_rolling_window:
        print(f"Warning: Not enough data or all NaNs for rolling STD {std_rolling_window} in Z-score calc for market {df['market'].iloc[0] if not df.empty else 'Unknown'}. Assigning NaNs.")
        df['ma_diff_zscore'] = np.nan
    else:
        mean_ma_diff_pct = df['ma_diff_pct'].rolling(window=std_rolling_window, min_periods=max(1, std_rolling_window // 2)).mean()
        std_ma_diff_pct = df['ma_diff_pct'].rolling(window=std_rolling_window, min_periods=max(1, std_rolling_window // 2)).std()
        df['ma_diff_zscore'] = (df['ma_diff_pct'] - mean_ma_diff_pct) / std_ma_diff_pct.replace(0, np.nan)
        df['ma_diff_zscore'].replace([np.inf, -np.inf], np.nan, inplace=True)
        
    df['delta_ma_diff_zscore'] = df['ma_diff_zscore'].diff()
    return df

def engineer_features_event_flow(df_events):
    df_eng = df_events.copy()
    if df_eng.empty:
        return df_eng

    df_eng['inst_flow_in_corrective_dir'] = df_eng['cumulative_inst_flow_in_window'] * np.sign(-df_eng['initial_ma_diff_pct'])
    df_eng['retail_flow_in_corrective_dir'] = df_eng['cumulative_retail_flow_in_window'] * np.sign(-df_eng['initial_ma_diff_pct'])

    df_eng['abs_initial_ma_diff_pct'] = np.abs(df_eng['initial_ma_diff_pct'])
    df_eng['abs_initial_ma_diff_zscore'] = np.abs(df_eng['initial_ma_diff_zscore'])


    df_eng['total_corrective_flow'] = df_eng['inst_flow_in_corrective_dir'] + df_eng['retail_flow_in_corrective_dir']

    epsilon = 1e-6
    df_eng['inst_to_total_corrective_ratio'] = df_eng['inst_flow_in_corrective_dir'] / (np.abs(df_eng['total_corrective_flow']) + epsilon)
    df_eng['retail_to_total_corrective_ratio'] = df_eng['retail_flow_in_corrective_dir'] / (np.abs(df_eng['total_corrective_flow']) + epsilon)
    df_eng['inst_to_retail_corrective_flow_ratio'] = df_eng['inst_flow_in_corrective_dir'] / (df_eng['retail_flow_in_corrective_dir'] + np.sign(df_eng['retail_flow_in_corrective_dir'].fillna(0)) * epsilon + epsilon)

    df_eng['inst_correction_effort_index'] = df_eng['inst_flow_in_corrective_dir'] / (df_eng['abs_initial_ma_diff_pct'] * 100 + epsilon)
    df_eng['retail_correction_effort_index'] = df_eng['retail_flow_in_corrective_dir'] / (df_eng['abs_initial_ma_diff_pct'] * 100 + epsilon)
    df_eng['total_correction_effort_index'] = df_eng['total_corrective_flow'] / (df_eng['abs_initial_ma_diff_pct'] * 100 + epsilon)
    
    return df_eng

def custom_describe(series, prefix=""):
    if series.empty:
        std_desc_keys = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        custom_keys = [f'{prefix}non_zero_count', f'{prefix}positive_count', f'{prefix}negative_count', f'{prefix}pct_positive', f'{prefix}pct_negative']
        all_keys = std_desc_keys + custom_keys
        nan_series = pd.Series(data=[np.nan]*len(all_keys), index=all_keys, name=series.name if series.name else "empty_series_description")
        return nan_series
        
    desc = series.describe()
    desc[f'{prefix}non_zero_count'] = (series.fillna(0) != 0).sum()
    desc[f'{prefix}positive_count'] = (series > 0).sum()
    desc[f'{prefix}negative_count'] = (series < 0).sum()
    
    count_val = desc.get('count', 0)
    if count_val > 0:
        desc[f'{prefix}pct_positive'] = desc[f'{prefix}positive_count'] / count_val
        desc[f'{prefix}pct_negative'] = desc[f'{prefix}negative_count'] / count_val
    else:
        desc[f'{prefix}pct_positive'] = 0.0
        desc[f'{prefix}pct_negative'] = 0.0
    return desc

def perform_adf_test_detailed(series, series_name=""):
    if series.empty or series.nunique(dropna=False) < 2: return 1.0, f"{series_name}: Series empty or constant."
    try:
        cleaned_series = series.dropna()
        if cleaned_series.empty or cleaned_series.nunique() < 2: return 1.0, f"{series_name}: Series empty or constant after dropna."
        result = adfuller(cleaned_series, regression='c', autolag='AIC'); p_value = result[1]
        return p_value, f"ADF for {series_name}: p={p_value:.4f}, n_obs={len(cleaned_series)}"
    except Exception as e: return 1.0, f"ADF test failed for {series_name}: {e}"

def make_series_stationary_for_granger(input_series, series_name="", max_diffs=1, sig_level=GRANGER_ADF_SIGNIFICANCE_LEVEL):
    current_series = input_series.copy() 
    is_stationary = False
    final_diff_count = 0
    stationarity_details = []

    for d_count in range(max_diffs + 1):
        if d_count > 0:
            current_series = current_series.diff() 
        
        series_to_test = current_series.dropna() 
        p_value, adf_msg = perform_adf_test_detailed(series_to_test, f"{series_name}_d{d_count}")
        stationarity_details.append({'series': f"{series_name}_d{d_count}", 'p_value': p_value, 'adf_msg': adf_msg, 'diff_order': d_count})
        
        if p_value <= sig_level:
            is_stationary = True
            final_diff_count = d_count
            break 
        if d_count == max_diffs and not is_stationary:
            final_diff_count = d_count

    return current_series.dropna(), final_diff_count, is_stationary, stationarity_details


def run_zscore_event_flow_study(df_timebars_full, markets_to_run):
    print("\n--- Part 1: Z-Score Event Flow Dynamics Study ---")
    all_market_event_flow_dynamics = []

    for market_name in tqdm(markets_to_run, desc="Processing Markets (Z-Score Event Flow)", unit="market"):
        df_market_orig = df_timebars_full[df_timebars_full['market'] == market_name].copy()
        if df_market_orig.empty:
            print(f"No data for market {market_name}. Skipping Z-score event flow study.")
            continue

        df_market_with_zscore = calculate_zscore_ma_diff(df_market_orig, ZSCORE_MA_WINDOW, ZSCORE_STD_ROLLING_WINDOW)
        
        if 'ma_diff_zscore' not in df_market_with_zscore.columns or df_market_with_zscore['ma_diff_zscore'].isna().all():
            print(f"Could not calculate ma_diff_zscore for {market_name}. Skipping.")
            continue

        event_condition = np.abs(df_market_with_zscore['ma_diff_zscore']) > ZSCORE_THRESHOLD
        mispricing_event_times = df_market_with_zscore.index[event_condition & event_condition.notna()]


        if mispricing_event_times.empty:
            print(f"No Z-score mispricing events found for {market_name} with threshold {ZSCORE_THRESHOLD}.")
            continue
        
        print(f"Found {len(mispricing_event_times)} Z-score mispricing events for {market_name}.")

        market_events_list = []
        for event_time_start_ts in tqdm(mispricing_event_times, desc=f"Z-Events ({market_name})", leave=False, position=1):
            initial_zscore = df_market_with_zscore.loc[event_time_start_ts, 'ma_diff_zscore']
            initial_ma_diff_pct_val = df_market_with_zscore.loc[event_time_start_ts, 'ma_diff_pct']
            
            if pd.isna(initial_zscore) or pd.isna(initial_ma_diff_pct_val): continue

            try:
                event_iloc = df_market_with_zscore.index.get_loc(event_time_start_ts)
            except KeyError: 
                print(f"Warning: Event time {event_time_start_ts} not found in index for {market_name} during event processing.")
                continue

            for window_len in ZSCORE_EVENT_FLOW_WINDOWS:
                segment_start_iloc = event_iloc + 1
                segment_end_iloc = segment_start_iloc + window_len
                if segment_end_iloc > len(df_market_with_zscore): continue
                
                event_window_df = df_market_with_zscore.iloc[segment_start_iloc:segment_end_iloc]
                if event_window_df.empty: continue

                cum_retail_flow = event_window_df['Retail'].sum()
                cum_inst_flow = event_window_df['Institutional'].sum()
                
                final_ma_diff_pct_val = event_window_df['ma_diff_pct'].iloc[-1]
                if pd.isna(final_ma_diff_pct_val): continue

                gap_closed_pct_val = np.nan
                if abs(initial_ma_diff_pct_val) > 1e-9:
                    gap_closed_pct_val = 1 - (abs(final_ma_diff_pct_val) / abs(initial_ma_diff_pct_val))
                
                is_successful = gap_closed_pct_val >= ZSCORE_SUCCESS_THRESHOLD_PCT if not pd.isna(gap_closed_pct_val) else False
                
                market_events_list.append({
                    'market': market_name,
                    'event_time_start': event_time_start_ts,
                    'window_length_cfg': window_len,
                    'initial_ma_diff_zscore': initial_zscore,
                    'initial_ma_diff_pct': initial_ma_diff_pct_val,
                    'final_ma_diff_pct': final_ma_diff_pct_val,
                    'gap_closed_pct': gap_closed_pct_val,
                    'is_successful_correction': is_successful,
                    'cumulative_retail_flow_in_window': cum_retail_flow,
                    'cumulative_inst_flow_in_window': cum_inst_flow,
                    'n_bars_in_window': len(event_window_df)
                })
        all_market_event_flow_dynamics.extend(market_events_list)

    if not all_market_event_flow_dynamics:
        print("No Z-score event flow dynamics data generated for any market.")
        return pd.DataFrame()

    df_event_flow_summary_zscore = pd.DataFrame(all_market_event_flow_dynamics)
    df_event_flow_summary_zscore = engineer_features_event_flow(df_event_flow_summary_zscore)
    
    output_path = ZSCORE_MAIN_OUTPUT_DIR / "event_flow_dynamics_summary_ZSCORE.csv"
    df_event_flow_summary_zscore.sort_values(['market', 'event_time_start', 'window_length_cfg']).to_csv(output_path, index=False)
    print(f"\nZ-Score Event flow dynamics summary saved to {output_path}")
    return df_event_flow_summary_zscore


def analyze_kh_puzzle_zscore(df_market_kh_zscore_events, results_container, output_dir):
    print("\n--- Analyzing KH Puzzle (Z-Score Events) ---")
    results_container['kh_puzzle_zscore'] = {} 

    if df_market_kh_zscore_events.empty:
        print("KH Z-score event data is empty. Skipping KH puzzle analysis.")
        return

    for wl_cfg in df_market_kh_zscore_events['window_length_cfg'].unique():
        print(f"\n  KH Market (Z-Score) - Window: {wl_cfg} min")
        df_kh_wl = df_market_kh_zscore_events[df_market_kh_zscore_events['window_length_cfg'] == wl_cfg].copy()
        
        kh_unsuccessful = df_kh_wl[df_kh_wl['is_successful_correction'] == False]
        kh_successful = df_kh_wl[df_kh_wl['is_successful_correction'] == True]

        if kh_unsuccessful.empty: 
            print(f"    No unsuccessful Z-score corrections for KH {wl_cfg} min window.")
            results_container['kh_puzzle_zscore'][f'wl_{wl_cfg}'] = {"status": f"No unsuccessful events for wl {wl_cfg}"}
            continue 
        
        current_wl_results = {} 
        
        desc_cols = ['abs_initial_ma_diff_zscore', 'abs_initial_ma_diff_pct', 'gap_closed_pct',
                     'inst_flow_in_corrective_dir', 'retail_flow_in_corrective_dir',
                     'total_corrective_flow', 'total_correction_effort_index',
                     'inst_correction_effort_index', 'retail_correction_effort_index']
        
        missing_cols_unsucc = [col for col in desc_cols if col not in kh_unsuccessful.columns]
        if missing_cols_unsucc:
            print(f"    Warning: Missing columns in kh_unsuccessful for wl {wl_cfg}: {missing_cols_unsucc}")
            current_desc_cols_unsucc = [col for col in desc_cols if col in kh_unsuccessful.columns]
        else:
            current_desc_cols_unsucc = desc_cols

        if current_desc_cols_unsucc:
            desc_kh_unsuccessful = kh_unsuccessful[current_desc_cols_unsucc].agg(lambda x: custom_describe(x))
            current_wl_results['unsuccessful_desc'] = desc_kh_unsuccessful
            try:
                desc_kh_unsuccessful.to_csv(output_dir / f"kh_zscore_wl_{wl_cfg}_unsuccessful_desc.csv")
            except Exception as e: print(f"Error saving kh_zscore_wl_{wl_cfg}_unsuccessful_desc.csv: {e}")


        if not kh_successful.empty:
            missing_cols_succ = [col for col in desc_cols if col not in kh_successful.columns]
            if missing_cols_succ:
                print(f"    Warning: Missing columns in kh_successful for wl {wl_cfg}: {missing_cols_succ}")
                current_desc_cols_succ = [col for col in desc_cols if col in kh_successful.columns]
            else:
                current_desc_cols_succ = desc_cols
            
            if current_desc_cols_succ:
                desc_kh_successful = kh_successful[current_desc_cols_succ].agg(lambda x: custom_describe(x))
                current_wl_results['successful_desc'] = desc_kh_successful
                try:
                    desc_kh_successful.to_csv(output_dir / f"kh_zscore_wl_{wl_cfg}_successful_desc.csv")
                except Exception as e: print(f"Error saving kh_zscore_wl_{wl_cfg}_successful_desc.csv: {e}")
        
        kh_unsuccessful_inst_trying = kh_unsuccessful[kh_unsuccessful['inst_flow_in_corrective_dir'] > 0]
        if not kh_unsuccessful_inst_trying.empty:
            missing_cols_inst_try = [col for col in desc_cols if col not in kh_unsuccessful_inst_trying.columns]
            current_desc_cols_inst_try = [col for col in desc_cols if col in kh_unsuccessful_inst_trying.columns]

            if current_desc_cols_inst_try:
                desc_kh_unsuccessful_inst_trying = kh_unsuccessful_inst_trying[current_desc_cols_inst_try].agg(lambda x: custom_describe(x))
                current_wl_results['unsuccessful_inst_trying_desc'] = desc_kh_unsuccessful_inst_trying
                try:
                    desc_kh_unsuccessful_inst_trying.to_csv(output_dir / f"kh_zscore_wl_{wl_cfg}_unsuccessful_inst_trying_desc.csv")
                except Exception as e: print(f"Error saving kh_zscore_wl_{wl_cfg}_unsuccessful_inst_trying_desc.csv: {e}")
            
            if len(kh_unsuccessful_inst_trying) > 1 and 'abs_initial_ma_diff_zscore' in kh_unsuccessful_inst_trying.columns and 'inst_flow_in_corrective_dir' in kh_unsuccessful_inst_trying.columns:
                col1 = kh_unsuccessful_inst_trying['abs_initial_ma_diff_zscore'].dropna() 
                col2 = kh_unsuccessful_inst_trying['inst_flow_in_corrective_dir'].dropna()
                common_idx = col1.index.intersection(col2.index)
                if len(common_idx) > 1:
                    corr, p_val = spearmanr(col1.loc[common_idx], col2.loc[common_idx])
                    corr_res = {'corr': corr, 'p_value': p_val, 'n_obs': len(common_idx)}
                    current_wl_results['corr_unsucc_inst_try_zscore_vs_instflow'] = corr_res
                    try:
                        pd.Series(corr_res).to_csv(output_dir / f"kh_zscore_wl_{wl_cfg}_corr_unsucc_inst_try_zscore_vs_instflow_params.csv", header=False)
                    except Exception as e: print(f"Error saving kh_zscore_wl_{wl_cfg}_corr_unsucc_inst_try_zscore_vs_instflow_params.csv: {e}")
        
        results_container['kh_puzzle_zscore'][f'wl_{wl_cfg}'] = current_wl_results
    print(f"KH Z-score puzzle detailed CSVs saving process attempted to {output_dir}")


def analyze_djt_dynamics_zscore(df_market_djt_zscore_events, results_container, output_dir):
    print("\n--- Analyzing DJT Dynamics (Z-Score Events) ---")
    results_container['djt_dynamics_zscore'] = {}

    if df_market_djt_zscore_events.empty:
        print("DJT Z-score event data is empty. Skipping DJT dynamics analysis.")
        return

    for wl_cfg in df_market_djt_zscore_events['window_length_cfg'].unique():
        print(f"\n  DJT Market (Z-Score) - Window: {wl_cfg} min")
        df_djt_wl = df_market_djt_zscore_events[df_market_djt_zscore_events['window_length_cfg'] == wl_cfg].copy()

        djt_unsuccessful = df_djt_wl[df_djt_wl['is_successful_correction'] == False]
        djt_successful = df_djt_wl[df_djt_wl['is_successful_correction'] == True]

        if djt_unsuccessful.empty: 
            print(f"    No unsuccessful Z-score corrections for DJT {wl_cfg} min window.")
            results_container['djt_dynamics_zscore'][f'wl_{wl_cfg}'] = {"status": f"No unsuccessful events for wl {wl_cfg}"}
            continue
        
        current_wl_results = {}
        desc_cols = ['abs_initial_ma_diff_zscore', 'abs_initial_ma_diff_pct', 'gap_closed_pct',
                     'inst_flow_in_corrective_dir', 'retail_flow_in_corrective_dir',
                     'total_corrective_flow', 'total_correction_effort_index',
                     'inst_correction_effort_index', 'retail_correction_effort_index']

        djt_unsuccessful_inst_counter = djt_unsuccessful[djt_unsuccessful['inst_flow_in_corrective_dir'] <= 0]
        if not djt_unsuccessful_inst_counter.empty:
            current_desc_cols_unsucc_counter = [col for col in desc_cols if col in djt_unsuccessful_inst_counter.columns]
            if current_desc_cols_unsucc_counter:
                desc_djt_unsucc_inst_counter = djt_unsuccessful_inst_counter[current_desc_cols_unsucc_counter].agg(lambda x: custom_describe(x))
                current_wl_results['unsuccessful_inst_counter_desc'] = desc_djt_unsucc_inst_counter
                try:
                    desc_djt_unsucc_inst_counter.to_csv(output_dir / f"djt_zscore_wl_{wl_cfg}_unsuccessful_inst_counter_desc.csv")
                except Exception as e: print(f"Error saving djt_zscore_wl_{wl_cfg}_unsuccessful_inst_counter_desc.csv: {e}")


            if len(djt_unsuccessful_inst_counter) > 1 and 'abs_initial_ma_diff_zscore' in djt_unsuccessful_inst_counter.columns and 'inst_flow_in_corrective_dir' in djt_unsuccessful_inst_counter.columns:
                col1 = djt_unsuccessful_inst_counter['abs_initial_ma_diff_zscore'].dropna() 
                col2 = djt_unsuccessful_inst_counter['inst_flow_in_corrective_dir'].dropna()
                common_idx = col1.index.intersection(col2.index)
                if len(common_idx) > 1:
                    corr, p_val = spearmanr(col1.loc[common_idx], col2.loc[common_idx])
                    corr_res = {'corr': corr, 'p_value': p_val, 'n_obs': len(common_idx)}
                    current_wl_results['corr_unsucc_inst_counter_zscore_vs_instflow'] = corr_res
                    try:
                        pd.Series(corr_res).to_csv(output_dir / f"djt_zscore_wl_{wl_cfg}_corr_unsucc_inst_counter_zscore_vs_instflow_params.csv", header=False)
                    except Exception as e: print(f"Error saving djt_zscore_wl_{wl_cfg}_corr_unsucc_inst_counter_zscore_vs_instflow_params.csv: {e}")
        
        if not djt_successful.empty:
            if 'inst_flow_in_corrective_dir' in djt_successful.columns:
                desc_djt_succ_inst_flow = custom_describe(djt_successful['inst_flow_in_corrective_dir'], "inst_flow_corrective_dir_")
                current_wl_results['successful_inst_flow_dist_desc'] = desc_djt_succ_inst_flow
                try:
                    desc_djt_succ_inst_flow.to_csv(output_dir / f"djt_zscore_wl_{wl_cfg}_successful_inst_flow_dist_desc.csv")
                except Exception as e: print(f"Error saving djt_zscore_wl_{wl_cfg}_successful_inst_flow_dist_desc.csv: {e}")


            djt_successful_inst_helping = djt_successful[djt_successful['inst_flow_in_corrective_dir'] > 0]
            if not djt_successful_inst_helping.empty:
                current_desc_cols_succ_help = [col for col in desc_cols if col in djt_successful_inst_helping.columns]
                if current_desc_cols_succ_help:
                    desc_temp = djt_successful_inst_helping[current_desc_cols_succ_help].agg(lambda x: custom_describe(x))
                    current_wl_results['successful_inst_helping_desc'] = desc_temp
                    try:
                        desc_temp.to_csv(output_dir / f"djt_zscore_wl_{wl_cfg}_successful_inst_helping_desc.csv")
                    except Exception as e: print(f"Error saving djt_zscore_wl_{wl_cfg}_successful_inst_helping_desc.csv: {e}")
            
            djt_successful_inst_hurting = djt_successful[djt_successful['inst_flow_in_corrective_dir'] < 0]
            if not djt_successful_inst_hurting.empty:
                current_desc_cols_succ_hurt = [col for col in desc_cols if col in djt_successful_inst_hurting.columns]
                if current_desc_cols_succ_hurt:
                    desc_temp = djt_successful_inst_hurting[current_desc_cols_succ_hurt].agg(lambda x: custom_describe(x))
                    current_wl_results['successful_inst_hurting_desc'] = desc_temp
                    try:
                        desc_temp.to_csv(output_dir / f"djt_zscore_wl_{wl_cfg}_successful_inst_hurting_desc.csv")
                    except Exception as e: print(f"Error saving djt_zscore_wl_{wl_cfg}_successful_inst_hurting_desc.csv: {e}")
        
        results_container['djt_dynamics_zscore'][f'wl_{wl_cfg}'] = current_wl_results
    print(f"DJT Z-score dynamics detailed CSVs saving process attempted to {output_dir}")


def run_granger_causality_zscore(df_timebars_full, markets_to_run, zscore_ma_window, zscore_std_window):
    print("\n--- Part 3 & 4: Granger Causality with Z-Score Gap ---")
    bivariate_granger_results_zscore = []
    event_based_granger_results_zscore = []
    stationarity_log_granger_zscore = []

    for market_name in tqdm(markets_to_run, desc="Granger (Z-Score Gap)", unit="market"):
        df_market_orig = df_timebars_full[df_timebars_full['market'] == market_name].copy()
        if df_market_orig.empty: continue

        df_market_with_zscore = calculate_zscore_ma_diff(df_market_orig, zscore_ma_window, zscore_std_window)
        if 'delta_ma_diff_zscore' not in df_market_with_zscore.columns or df_market_with_zscore['delta_ma_diff_zscore'].isna().all():
            print(f"Skipping Granger for {market_name} due to missing delta_ma_diff_zscore.")
            continue
            
        df_market_with_zscore['deltaP'] = df_market_with_zscore['price'].diff()
        df_market_with_zscore['delta_RetailFlow'] = df_market_with_zscore['Retail'].diff()
        df_market_with_zscore['delta_InstFlow'] = df_market_with_zscore['Institutional'].diff()

        series_to_test_bivar = {
            'deltaP': df_market_with_zscore['deltaP'],
            'delta_RetailFlow': df_market_with_zscore['delta_RetailFlow'],
            'delta_InstFlow': df_market_with_zscore['delta_InstFlow'],
            'delta_ma_diff_zscore': df_market_with_zscore['delta_ma_diff_zscore']
        }
        
        stationary_series_bivar = {}
        all_bivar_series_valid = True
        for key, series_data in series_to_test_bivar.items():
            if series_data.isna().all():
                print(f"Warning: Series {key} for {market_name} (Z-score Granger) is all NaN before stationarity check.")
                all_bivar_series_valid = False; break
            s_series, d_count, is_stat, stat_log = make_series_stationary_for_granger(series_data, f"{market_name}_{key}_ZGap")
            stationary_series_bivar[key] = s_series
            stationarity_log_granger_zscore.extend(stat_log)
            if not is_stat: 
                print(f"Warning: {key} for {market_name} (Z-score Gap Granger) not stationary after differencing.")
        if not all_bivar_series_valid: continue


        pairs_to_test = [
            (['deltaP', 'delta_RetailFlow'], 'deltaRetailFlow_to_deltaP_ZGap', 'deltaP_to_deltaRetailFlow_ZGap'),
            (['delta_ma_diff_zscore', 'delta_InstFlow'], 'deltaInstFlow_to_deltaMaDiffZscore', 'deltaMaDiffZscore_to_deltaInstFlow')
        ]

        for (var_keys_for_pair, test_name_fwd, test_name_rev) in pairs_to_test:
            if var_keys_for_pair[0] not in stationary_series_bivar or var_keys_for_pair[1] not in stationary_series_bivar or \
               stationary_series_bivar[var_keys_for_pair[0]].isna().all() or stationary_series_bivar[var_keys_for_pair[1]].isna().all():
                print(f"Skipping Granger pair {test_name_fwd} for {market_name} due to missing/all-NaN stationary series.")
                continue

            df_pair_granger = pd.concat([stationary_series_bivar[var_keys_for_pair[0]], stationary_series_bivar[var_keys_for_pair[1]]], axis=1).dropna()
            
            if len(df_pair_granger) > max(GRANGER_LAGS_TO_TEST) + 20 and df_pair_granger.shape[1] == 2:
                try:
                    gc_res_fwd = grangercausalitytests(df_pair_granger[[var_keys_for_pair[0], var_keys_for_pair[1]]], GRANGER_LAGS_TO_TEST, verbose=False)
                    for lag in GRANGER_LAGS_TO_TEST:
                        if lag in gc_res_fwd: bivariate_granger_results_zscore.append({'market': market_name, 'test': test_name_fwd, 'lag': lag, 'F_stat': gc_res_fwd[lag][0]['ssr_ftest'][0], 'p_value': gc_res_fwd[lag][0]['ssr_ftest'][1]})
                    
                    gc_res_rev = grangercausalitytests(df_pair_granger[[var_keys_for_pair[1], var_keys_for_pair[0]]], GRANGER_LAGS_TO_TEST, verbose=False) 
                    for lag in GRANGER_LAGS_TO_TEST:
                        if lag in gc_res_rev: bivariate_granger_results_zscore.append({'market': market_name, 'test': test_name_rev, 'lag': lag, 'F_stat': gc_res_rev[lag][0]['ssr_ftest'][0], 'p_value': gc_res_rev[lag][0]['ssr_ftest'][1]})
                except Exception as e: print(f"Error in Bivariate Granger for {test_name_fwd}/{test_name_rev} for {market_name} (Z-score): {e}")
        
        event_condition_eb = np.abs(df_market_with_zscore['ma_diff_zscore']) > ZSCORE_THRESHOLD
        mispricing_event_times_eb = df_market_with_zscore.index[event_condition_eb & event_condition_eb.notna()]

        if not mispricing_event_times_eb.empty:
            for event_time_start_ts_eb in tqdm(mispricing_event_times_eb, desc=f"EventGranger ({market_name}, ZGap)", leave=False, position=1):
                try: event_iloc_eb = df_market_with_zscore.index.get_loc(event_time_start_ts_eb)
                except KeyError: continue

                for window_len_eb in ZSCORE_EVENT_FLOW_WINDOWS: 
                    segment_start_iloc_eb = event_iloc_eb + 1
                    segment_end_iloc_eb = segment_start_iloc_eb + window_len_eb
                    if segment_end_iloc_eb > len(df_market_with_zscore): continue
                    
                    event_segment_raw = df_market_with_zscore.iloc[segment_start_iloc_eb:segment_end_iloc_eb][['delta_InstFlow', 'delta_ma_diff_zscore']].copy()
                    
                    if event_segment_raw['delta_InstFlow'].isna().all() or event_segment_raw['delta_ma_diff_zscore'].isna().all():
                        continue

                    s_if_ev, _, _, _ = make_series_stationary_for_granger(event_segment_raw['delta_InstFlow'], f"{market_name}_EvDeltaInstFlow_ZGap_Temp")
                    s_dmdz_ev, _, _, _ = make_series_stationary_for_granger(event_segment_raw['delta_ma_diff_zscore'], f"{market_name}_EvDeltaMaDiffZscore_Temp")
                    
                    df_ev_stat = pd.concat([s_dmdz_ev, s_if_ev], axis=1).dropna() 
                    
                    if len(df_ev_stat) >= GRANGER_EVENT_MIN_OBS_FOR_TEST and df_ev_stat.shape[1] == 2:
                        max_dynamic_lag = max(1, int(len(df_ev_stat) / 3) -1)
                        lags_to_test_event = [l for l in GRANGER_LAGS_TO_TEST if l <= max_dynamic_lag]
                        if not lags_to_test_event: lags_to_test_event = [1] if 1 <= max_dynamic_lag else []
                        if not lags_to_test_event: continue
                        
                        try:
                            gc_res_ev = grangercausalitytests(df_ev_stat, lags_to_test_event, verbose=False)
                            for lag_ev in lags_to_test_event:
                                if lag_ev in gc_res_ev and gc_res_ev[lag_ev]:
                                    test_output = gc_res_ev[lag_ev][0]
                                    if 'ssr_ftest' in test_output and test_output['ssr_ftest'] is not None:
                                        event_based_granger_results_zscore.append({
                                            'market': market_name, 'event_time': event_time_start_ts_eb, 'window_length': window_len_eb,
                                            'test': 'deltaInstFlow_to_deltaMaDiffZscore_EVENT', 'lag': lag_ev,
                                            'F_stat': test_output['ssr_ftest'][0], 'p_value': test_output['ssr_ftest'][1],
                                            'n_obs_segment': len(df_ev_stat)
                                        })
                        except Exception as e: 
                            pass 

    if stationarity_log_granger_zscore:
        pd.DataFrame(stationarity_log_granger_zscore).to_csv(ZSCORE_GRANGER_OUTPUT_DIR / "stationarity_log_granger_ZSCORE.csv", index=False)
    if bivariate_granger_results_zscore:
        pd.DataFrame(bivariate_granger_results_zscore).sort_values(['market', 'test', 'lag']).to_csv(ZSCORE_GRANGER_OUTPUT_DIR / "bivariate_granger_summary_ZSCORE.csv", index=False)
    if event_based_granger_results_zscore:
        pd.DataFrame(event_based_granger_results_zscore).sort_values(['market', 'event_time', 'lag']).to_csv(ZSCORE_GRANGER_OUTPUT_DIR / "event_based_granger_summary_ZSCORE.csv", index=False)
    print(f"Z-Score Granger causality results saving process attempted to {ZSCORE_GRANGER_OUTPUT_DIR}")


def main():
    print(f"Starting Z-Score Deep Dive Analysis - {datetime.datetime.now()}")
    
    try:
        df_timebars_full = pd.read_parquet(TIMEBARS_FILE_PATH)
        if 'time' not in df_timebars_full.columns: raise ValueError("'time' column missing.")
        df_timebars_full['time'] = pd.to_datetime(df_timebars_full['time'])
        df_timebars_full.set_index('time', inplace=True)
        df_timebars_full.sort_index(inplace=True)
        print(f"Successfully loaded and processed timebars.parquet: {len(df_timebars_full)} rows.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load or process {TIMEBARS_FILE_PATH}. Exiting. Error: {e}")
        return

    markets = df_timebars_full['market'].unique()
    if len(markets) == 0:
        print("No markets found in timebars data. Exiting.")
        return

    df_zscore_event_summary = run_zscore_event_flow_study(df_timebars_full, markets)

    if df_zscore_event_summary is not None and not df_zscore_event_summary.empty:
        print("\n--- Part 2: Detailed Segmentation Analysis of Z-Score Events ---")
        all_detailed_segment_results = {} 

        df_kh_zscore_events = df_zscore_event_summary[df_zscore_event_summary['market'] == 'PRES-2024-KH'].copy()
        if not df_kh_zscore_events.empty:
            analyze_kh_puzzle_zscore(df_kh_zscore_events, all_detailed_segment_results, ZSCORE_DETAILED_SEGMENT_REPORTS_DIR)
        else:
            print("No Z-score event data for PRES-2024-KH to segment.")

        df_djt_zscore_events = df_zscore_event_summary[df_zscore_event_summary['market'] == 'PRES-2024-DJT'].copy()
        if not df_djt_zscore_events.empty:
            analyze_djt_dynamics_zscore(df_djt_zscore_events, all_detailed_segment_results, ZSCORE_DETAILED_SEGMENT_REPORTS_DIR)
        else:
            print("No Z-score event data for PRES-2024-DJT to segment.")
    else:
        print("Skipping detailed segmentation as Z-score event summary is empty or failed.")

    run_granger_causality_zscore(df_timebars_full, markets, ZSCORE_MA_WINDOW, ZSCORE_STD_ROLLING_WINDOW)

    print(f"\n--- Z-Score Deep Dive Analysis Script Finished - {datetime.datetime.now()} ---")
    print(f"All Z-score analysis output files saved in subdirectories of: {ZSCORE_MAIN_OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()