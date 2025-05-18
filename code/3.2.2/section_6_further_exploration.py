"""
analyzes event flow dynamics from a csv, engineering features, performs market-specific deep dives and saves results.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from tqdm import tqdm
import warnings

INPUT_FILE_PATH = Path("./event_flow_dynamics_summary.csv")
DEEP_DIVE_OUTPUT_DIR = Path("./")
DEEP_DIVE_OUTPUT_DIR.mkdir(exist_ok=True)

ALPHA = 0.05

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def engineer_features(df):
    df_eng = df.copy()

    df_eng['inst_flow_in_corrective_dir'] = df_eng['cumulative_inst_flow_in_window'] * np.sign(-df_eng['initial_ma_diff'])
    df_eng['retail_flow_in_corrective_dir'] = df_eng['cumulative_retail_flow_in_window'] * np.sign(-df_eng['initial_ma_diff'])

    df_eng['abs_initial_ma_diff'] = np.abs(df_eng['initial_ma_diff'])

    df_eng['total_corrective_flow'] = df_eng['inst_flow_in_corrective_dir'] + df_eng['retail_flow_in_corrective_dir']

    epsilon = 1e-6
    df_eng['inst_to_total_corrective_ratio'] = df_eng['inst_flow_in_corrective_dir'] / (np.abs(df_eng['total_corrective_flow']) + epsilon)
    df_eng['retail_to_total_corrective_ratio'] = df_eng['retail_flow_in_corrective_dir'] / (np.abs(df_eng['total_corrective_flow']) + epsilon)
    
    df_eng['inst_to_retail_corrective_flow_ratio'] = df_eng['inst_flow_in_corrective_dir'] / (df_eng['retail_flow_in_corrective_dir'] + np.sign(df_eng['retail_flow_in_corrective_dir']) * epsilon + epsilon)

    df_eng['correction_effort_index'] = df_eng['total_corrective_flow'] / (df_eng['abs_initial_ma_diff'] * 100 + epsilon)
    df_eng['inst_correction_effort_index'] = df_eng['inst_flow_in_corrective_dir'] / (df_eng['abs_initial_ma_diff'] * 100 + epsilon)
    df_eng['retail_correction_effort_index'] = df_eng['retail_flow_in_corrective_dir'] / (df_eng['abs_initial_ma_diff'] * 100 + epsilon)

    return df_eng

def custom_describe(series, prefix=""):
    desc = series.describe()
    desc[f'{prefix}non_zero_count'] = (series != 0).sum()
    desc[f'{prefix}positive_count'] = (series > 0).sum()
    desc[f'{prefix}negative_count'] = (series < 0).sum()
    if f'{prefix}count' in desc:
        desc[f'{prefix}pct_positive'] = desc[f'{prefix}positive_count'] / desc[f'{prefix}count'] if desc[f'{prefix}count'] > 0 else 0
        desc[f'{prefix}pct_negative'] = desc[f'{prefix}negative_count'] / desc[f'{prefix}count'] if desc[f'{prefix}count'] > 0 else 0
    return desc

def analyze_kh_puzzle(df_market_kh, results_container):
    results_container['kh_puzzle'] = {}

    for wl_cfg in df_market_kh['window_length_cfg'].unique():
        df_kh_wl = df_market_kh[df_market_kh['window_length_cfg'] == wl_cfg].copy()
        
        kh_unsuccessful = df_kh_wl[df_kh_wl['is_successful_correction'] == False]
        kh_successful = df_kh_wl[df_kh_wl['is_successful_correction'] == True]

        if kh_unsuccessful.empty:
            print(f"    No unsuccessful corrections for KH {wl_cfg} min window.")
            continue
        if kh_successful.empty:
            print(f"    No successful corrections for KH {wl_cfg} min window (for comparison).")
            
        results_container['kh_puzzle'][f'wl_{wl_cfg}'] = {}
        
        desc_kh_unsuccessful = kh_unsuccessful[['abs_initial_ma_diff', 'gap_closed_pct',
                                                'inst_flow_in_corrective_dir', 'retail_flow_in_corrective_dir',
                                                'total_corrective_flow', 'correction_effort_index',
                                                'inst_correction_effort_index', 'retail_correction_effort_index']].agg(
                                                    lambda x: custom_describe(x)
                                                )
        results_container['kh_puzzle'][f'wl_{wl_cfg}']['unsuccessful_desc'] = desc_kh_unsuccessful

        if not kh_successful.empty:
            desc_kh_successful = kh_successful[['abs_initial_ma_diff', 'gap_closed_pct',
                                                'inst_flow_in_corrective_dir', 'retail_flow_in_corrective_dir',
                                                'total_corrective_flow', 'correction_effort_index',
                                                'inst_correction_effort_index', 'retail_correction_effort_index']].agg(
                                                    lambda x: custom_describe(x)
                                                )
            results_container['kh_puzzle'][f'wl_{wl_cfg}']['successful_desc'] = desc_kh_successful
        
        kh_unsuccessful_inst_trying = kh_unsuccessful[kh_unsuccessful['inst_flow_in_corrective_dir'] > 0]
        if not kh_unsuccessful_inst_trying.empty:
            desc_kh_unsuccessful_inst_trying = kh_unsuccessful_inst_trying[
                ['abs_initial_ma_diff', 'gap_closed_pct', 'inst_flow_in_corrective_dir', 
                 'retail_flow_in_corrective_dir', 'total_corrective_flow', 'correction_effort_index']
            ].agg(lambda x: custom_describe(x))
            results_container['kh_puzzle'][f'wl_{wl_cfg}']['unsuccessful_inst_trying_desc'] = desc_kh_unsuccessful_inst_trying
            
            if len(kh_unsuccessful_inst_trying) > 1:
                corr_misprice_instflow, p_misprice_instflow = spearmanr(
                    kh_unsuccessful_inst_trying['abs_initial_ma_diff'],
                    kh_unsuccessful_inst_trying['inst_flow_in_corrective_dir']
                )
                results_container['kh_puzzle'][f'wl_{wl_cfg}']['corr_unsucc_inst_try_misprice_vs_instflow'] = {'corr': corr_misprice_instflow, 'p_value': p_misprice_instflow}

        kh_unsuccessful_retail_corrective = kh_unsuccessful[kh_unsuccessful['retail_flow_in_corrective_dir'] > 0]
        kh_unsuccessful_retail_counter = kh_unsuccessful[kh_unsuccessful['retail_flow_in_corrective_dir'] <= 0]

        if not kh_unsuccessful_retail_corrective.empty:
            desc_temp = kh_unsuccessful_retail_corrective[['inst_flow_in_corrective_dir', 'abs_initial_ma_diff']].agg(lambda x: custom_describe(x))
            results_container['kh_puzzle'][f'wl_{wl_cfg}']['unsuccessful_retail_corrective_desc'] = desc_temp
        if not kh_unsuccessful_retail_counter.empty:
            desc_temp = kh_unsuccessful_retail_counter[['inst_flow_in_corrective_dir', 'abs_initial_ma_diff']].agg(lambda x: custom_describe(x))
            results_container['kh_puzzle'][f'wl_{wl_cfg}']['unsuccessful_retail_counter_desc'] = desc_temp

def analyze_djt_dynamics(df_market_djt, results_container):
    results_container['djt_dynamics'] = {}

    for wl_cfg in df_market_djt['window_length_cfg'].unique():
        df_djt_wl = df_market_djt[df_market_djt['window_length_cfg'] == wl_cfg].copy()

        djt_unsuccessful = df_djt_wl[df_djt_wl['is_successful_correction'] == False]
        djt_successful = df_djt_wl[df_djt_wl['is_successful_correction'] == True]

        if djt_unsuccessful.empty: 
            print(f"    No unsuccessful corrections for DJT {wl_cfg} min window.")
            continue
        results_container['djt_dynamics'][f'wl_{wl_cfg}'] = {}

        djt_unsuccessful_inst_counter = djt_unsuccessful[djt_unsuccessful['inst_flow_in_corrective_dir'] <= 0]
        if not djt_unsuccessful_inst_counter.empty:
            desc_djt_unsucc_inst_counter = djt_unsuccessful_inst_counter[
                ['abs_initial_ma_diff', 'gap_closed_pct', 'inst_flow_in_corrective_dir', 
                 'retail_flow_in_corrective_dir', 'total_corrective_flow', 'correction_effort_index']
            ].agg(lambda x: custom_describe(x))
            results_container['djt_dynamics'][f'wl_{wl_cfg}']['unsuccessful_inst_counter_desc'] = desc_djt_unsucc_inst_counter

            if len(djt_unsuccessful_inst_counter) > 1:
                corr_misprice_instcounter, p_misprice_instcounter = spearmanr(
                    djt_unsuccessful_inst_counter['abs_initial_ma_diff'],
                    djt_unsuccessful_inst_counter['inst_flow_in_corrective_dir'] 
                )
                results_container['djt_dynamics'][f'wl_{wl_cfg}']['corr_unsucc_inst_counter_misprice_vs_instflow'] = {'corr': corr_misprice_instcounter, 'p_value': p_misprice_instcounter}
        
        if not djt_successful.empty:
            desc_djt_succ_inst_flow = custom_describe(djt_successful['inst_flow_in_corrective_dir'], "inst_flow_corrective_dir_")
            results_container['djt_dynamics'][f'wl_{wl_cfg}']['successful_inst_flow_dist_desc'] = desc_djt_succ_inst_flow

            djt_successful_inst_helping = djt_successful[djt_successful['inst_flow_in_corrective_dir'] > 0]
            if not djt_successful_inst_helping.empty:
                desc_temp = djt_successful_inst_helping[['gap_closed_pct', 'inst_flow_in_corrective_dir', 'abs_initial_ma_diff']].agg(lambda x: custom_describe(x))
                results_container['djt_dynamics'][f'wl_{wl_cfg}']['successful_inst_helping_desc'] = desc_temp
            
            djt_successful_inst_hurting = djt_successful[djt_successful['inst_flow_in_corrective_dir'] < 0] 
            if not djt_successful_inst_hurting.empty:
                desc_temp = djt_successful_inst_hurting[['gap_closed_pct', 'inst_flow_in_corrective_dir', 'abs_initial_ma_diff']].agg(lambda x: custom_describe(x))
                results_container['djt_dynamics'][f'wl_{wl_cfg}']['successful_inst_hurting_desc'] = desc_temp

def save_results(results_container, base_dir):
    for category_key, category_data in results_container.items():
        if isinstance(category_data, dict):
            for wl_key, wl_data in category_data.items(): 
                if isinstance(wl_data, dict):
                    for desc_key, data_item in wl_data.items(): 
                        filename_base = base_dir / f"{category_key}_{wl_key}_{desc_key}"
                        try:
                            if isinstance(data_item, pd.DataFrame) or isinstance(data_item, pd.Series):
                                data_item.to_csv(f"{filename_base}.csv")
                            elif isinstance(data_item, dict): 
                                pd.Series(data_item).to_csv(f"{filename_base}_params.csv")
                        except Exception as e:
                            print(f"Error saving {filename_base}: {e}")
    print(f"File saving process attempted. Check directory: {base_dir.resolve()}")

def main():
    print(f"Starting Deep Dive Analysis on: {INPUT_FILE_PATH}")
    try:
        df_orig = pd.read_csv(INPUT_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE_PATH}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    if df_orig.empty:
        print("Input DataFrame is empty. Exiting.")
        return
        
    print(f"Successfully loaded {len(df_orig)} records.")

    df_processed = engineer_features(df_orig)
    print("Successfully engineered features.")

    all_results = {}

    df_kh = df_processed[df_processed['market'] == 'PRES-2024-KH'].copy()
    df_djt = df_processed[df_processed['market'] == 'PRES-2024-DJT'].copy()

    if not df_kh.empty:
        analyze_kh_puzzle(df_kh, all_results)
    else:
        print("No data for PRES-2024-KH market.")
    
    if not df_djt.empty:
        analyze_djt_dynamics(df_djt, all_results)
    else:
        print("No data for PRES-2024-DJT market.")

    save_results(all_results, DEEP_DIVE_OUTPUT_DIR)
    
    print(f"\nDeep Dive Analysis Script Finished.")
    print(f"All output files saved in: {DEEP_DIVE_OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()