"""
analyzes event flow dynamics from a csv, computes statistics, and saves reports.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
from tqdm import tqdm

INPUT_FILE_PATH = Path("./event_flow_dynamics_summary.csv")
OUTPUT_DIR = Path("./")
OUTPUT_DIR.mkdir(exist_ok=True)

ALPHA = 0.05

def analyze_event_flow_dynamics():
    try:
        df = pd.read_csv(INPUT_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE_PATH}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    numeric_cols = ['initial_ma_diff', 'final_ma_diff', 'gap_closed_pct',
                    'cumulative_retail_flow_in_window', 'cumulative_inst_flow_in_window']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    critical_cols_for_nan_check = ['initial_ma_diff', 'cumulative_retail_flow_in_window', 'cumulative_inst_flow_in_window', 'is_successful_correction']
    df.dropna(subset=critical_cols_for_nan_check, inplace=True)

    df['inst_flow_in_corrective_dir'] = df['cumulative_inst_flow_in_window'] * np.sign(-df['initial_ma_diff'])
    df['retail_flow_in_corrective_dir'] = df['cumulative_retail_flow_in_window'] * np.sign(-df['initial_ma_diff'])
    
    overall_desc = df[numeric_cols + ['inst_flow_in_corrective_dir', 'retail_flow_in_corrective_dir']].describe()
    print(overall_desc)
    overall_desc.to_csv(OUTPUT_DIR / "overall_descriptive_stats.csv")

    significance_tests_list = []

    aggregations = {
        'initial_ma_diff': ['mean', 'median', 'std'],
        'gap_closed_pct': ['mean', 'median', 'std'],
        'cumulative_retail_flow_in_window': ['mean', 'median', 'std'],
        'cumulative_inst_flow_in_window': ['mean', 'median', 'std'],
        'retail_flow_in_corrective_dir': ['mean', 'median', 'std'],
        'inst_flow_in_corrective_dir': ['mean', 'median', 'std'],
        'event_time_start': ['count'] 
    }

    grouped = df.groupby(['market', 'window_length_cfg', 'is_successful_correction'])
    summary_stats = grouped.agg(aggregations)
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values] 
    summary_stats.rename(columns={'event_time_start_count': 'n_events'}, inplace=True)
    
    print(summary_stats)
    summary_stats.to_csv(OUTPUT_DIR / "grouped_summary_statistics.csv")

    for (market, window_len), group_df in tqdm(df.groupby(['market', 'window_length_cfg']), desc="Market/Window Tests"):
        successful_events = group_df[group_df['is_successful_correction'] == True]
        unsuccessful_events = group_df[group_df['is_successful_correction'] == False]

        if len(successful_events) < 5 or len(unsuccessful_events) < 5: 
            print(f"Skipping tests for {market}, Window {window_len} due to insufficient samples in one/both groups.")
            continue

        flow_vars_to_test = {
            'cumulative_inst_flow_in_window': 'Cumulative Institutional Flow',
            'inst_flow_in_corrective_dir': 'Institutional Flow in Corrective Direction',
            'cumulative_retail_flow_in_window': 'Cumulative Retail Flow',
            'retail_flow_in_corrective_dir': 'Retail Flow in Corrective Direction'
        }

        for var_key, var_name in flow_vars_to_test.items():
            try:
                stat, p_value = mannwhitneyu(successful_events[var_key].dropna(),
                                             unsuccessful_events[var_key].dropna(),
                                             alternative='two-sided') 
                
                significance_tests_list.append({
                    'market': market,
                    'window_length_cfg': window_len,
                    'variable_tested': var_key,
                    'variable_description': var_name,
                    'mannwhitneyu_statistic': stat,
                    'p_value': p_value,
                    'is_significant': p_value < ALPHA,
                    'n_successful': len(successful_events[var_key].dropna()),
                    'n_unsuccessful': len(unsuccessful_events[var_key].dropna()),
                    'median_successful': successful_events[var_key].median(),
                    'median_unsuccessful': unsuccessful_events[var_key].median()
                })
            except ValueError as ve: 
                 print(f"  Skipping test for {var_name} in {market}, Window {window_len} due to: {ve}")
            except Exception as e:
                print(f"  Error during Mann-Whitney U for {var_name} in {market}, Window {window_len}: {e}")


    if significance_tests_list:
        df_sig_tests = pd.DataFrame(significance_tests_list)
        df_sig_tests.sort_values(['market', 'window_length_cfg', 'p_value'], inplace=True)
        print(df_sig_tests)
        df_sig_tests.to_csv(OUTPUT_DIR / "flow_significance_tests.csv", index=False)
        
        print("\nKey Significant Findings (p < 0.05):")
        print(df_sig_tests[df_sig_tests['is_significant']])
    else:
        print("No significance tests were run (possibly due to insufficient group sizes).")

    print(f"All output files saved in: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    analyze_event_flow_dynamics()