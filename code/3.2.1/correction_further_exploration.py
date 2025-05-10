"""
analyzes data to identify retail-driven mispricing events and subsequent institutional correction dynamics across various parameter configurations
"""
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm
from scipy.stats import mannwhitneyu, pearsonr
from collections import Counter

base_out_dir = Path("./")
base_out_dir.mkdir(exist_ok=True)

print("Loading data...")
trades_master_df = pd.read_parquet("clean_trades.parquet")
trades_master_df["time"] = pd.to_datetime(trades_master_df["time"])
trades_master_df = trades_master_df.sort_values("time").reset_index(drop=True)

tb_master_df = pd.read_parquet("timebars.parquet")
tb_master_df["time"] = pd.to_datetime(tb_master_df["time"])
tb_master_df = tb_master_df.sort_values("time").reset_index(drop=True)
print("Data loaded.")

markets_to_analyze = ['PRES-2024-DJT', 'PRES-2024-KH']
if not all(m in trades_master_df["market"].unique() for m in markets_to_analyze):
    print(f"Warning: Not all specified markets {markets_to_analyze} found in data. Available: {trades_master_df['market'].unique()}")
    markets_to_analyze = [m for m in markets_to_analyze if m in trades_master_df["market"].unique()]

ma_window_options = [5, 15]
ma_diff_calculation_methods = ["percentage", "z_score"]
ma_thr_pct_default = 0.02
ma_thr_z_default = 2.0
z_score_rolling_window = 60

pre_window_duration_options = [timedelta(days=1), timedelta(hours=12)]
post_window_duration_options = [timedelta(minutes=30), timedelta(hours=1), timedelta(hours=2)]

cnt_thr_default = 0.50
flow_thrs_default = [0.05, 0.10, 0.20]

reversion_tol_default = 0.01
min_events_for_correlation = 10
max_detailed_events_to_log_per_config = 2
num_first_inst_trades_to_analyze = 3

all_market_run_records = []
all_market_run_lines = []
all_market_detailed_events_data = []

baseline_inst_shares_by_market = trades_master_df.groupby('market')["size"].apply(
    lambda x: trades_master_df.loc[x.index[trades_master_df.loc[x.index, "label"] == "Institutional"], "size"].sum() / x.sum()
).to_dict()

for market_name in markets_to_analyze:
    market_out_dir = base_out_dir / market_name
    market_out_dir.mkdir(exist_ok=True)

    report_path = market_out_dir / f"{market_name}_correction_further_report.txt"
    table_path = market_out_dir / f"{market_name}_correction_further_metrics.csv"
    detailed_events_log_path = market_out_dir / f"{market_name}_detailed_event_examples_log.csv"

    trades_current_market_df = trades_master_df[trades_master_df["market"] == market_name].copy()
    tb_current_market_df = tb_master_df[tb_master_df["market"] == market_name].copy()

    if trades_current_market_df.empty or tb_current_market_df.empty:
        print(f"No data for market {market_name}, skipping.")
        all_market_run_lines.append(f"\n\n===== Market: {market_name} =====\nNo data available.\n")
        continue

    t_idx_global = pd.DatetimeIndex(trades_current_market_df["time"])
    t_sz_global = trades_current_market_df["size"].values
    t_lbl_global = trades_current_market_df["label"].values
    t_px_global = trades_current_market_df["price"].values
    t_yes_global = np.where(t_px_global <= 50, (100 - t_px_global) / 100, t_px_global / 100)

    current_market_baseline_inst_share = baseline_inst_shares_by_market.get(market_name, np.nan)

    market_specific_records = []
    market_specific_lines = []
    market_specific_detailed_events = []
    market_specific_zscore_event_hours = []

    experiment_configs_inner = []
    for ma_win_opt in ma_window_options:
        for ma_diff_method_opt in ma_diff_calculation_methods:
            for pre_window_opt in pre_window_duration_options:
                for post_window_opt in post_window_duration_options:
                    experiment_configs_inner.append({
                        "ma_window": ma_win_opt,
                        "ma_diff_method": ma_diff_method_opt,
                        "pre_window_duration": pre_window_opt,
                        "post_window_duration": post_window_opt
                    })
    
    market_specific_lines.append(f"\n\n===== Market: {market_name} =====")
    market_specific_lines.append(f"Baseline overall institutional flow-share: {current_market_baseline_inst_share:.3%}")

    for exp_config in tqdm(experiment_configs_inner, desc=f"Experiments for {market_name}"):
        ma_win = exp_config["ma_window"]
        ma_diff_method = exp_config["ma_diff_method"]
        current_pre_window_duration = exp_config["pre_window_duration"]
        current_post_window_duration = exp_config["post_window_duration"]

        tb = tb_current_market_df.copy()
        ma_col_name = f"ma{ma_win}"
        tb[ma_col_name] = tb["price"].rolling(ma_win, min_periods=max(1, ma_win//2)).mean().bfill().ffill()

        current_ma_detection_threshold = np.nan
        if ma_diff_method == "percentage":
            tb["ma_diff_current"] = ((tb["price"] - tb[ma_col_name]) / tb[ma_col_name].replace(0, np.nan)).fillna(0)
            current_ma_detection_threshold = ma_thr_pct_default
        elif ma_diff_method == "z_score":
            ma_diff_raw_pct = ((tb["price"] - tb[ma_col_name]) / tb[ma_col_name].replace(0, np.nan)).fillna(0)
            ma_diff_rolling_std = ma_diff_raw_pct.rolling(window=z_score_rolling_window, min_periods=max(1, z_score_rolling_window//2)).std().bfill().ffill()
            tb["ma_diff_current"] = (ma_diff_raw_pct / ma_diff_rolling_std.replace(0, np.nan)).fillna(0)
            current_ma_detection_threshold = ma_thr_z_default
        
        tb_idx_current = pd.DatetimeIndex(tb["time"])
        tb_px_current_np = tb["price"].values
        tb_yes_current_np = np.where(tb_px_current_np <= 50, (100 - tb_px_current_np) / 100, tb_px_current_np / 100)
        tb_ma_diff_current_np = tb["ma_diff_current"].values
        tb_ma_current_np = tb[ma_col_name].values

        retail_definition_configs = [("count", cnt_thr_default)] + [("flow", thr) for thr in flow_thrs_default]
        
        for retail_typ, retail_thr in retail_definition_configs:
            events_detected_for_detailed_log = 0
            ev_idx = np.where(np.abs(tb_ma_diff_current_np) > current_ma_detection_threshold)[0]
            
            ev, at, su = 0, 0, 0
            inst_succ_shares, inst_fail_shares = [], []
            inst_all_shares_post_event, gap_closed_all_events = [], []
            first_N_inst_trade_impacts_succ = [[] for _ in range(num_first_inst_trades_to_analyze)]
            first_N_inst_trade_impacts_fail = [[] for _ in range(num_first_inst_trades_to_analyze)]
            first_N_inst_trade_correct_dir_succ = [[] for _ in range(num_first_inst_trades_to_analyze)]
            first_N_inst_trade_correct_dir_fail = [[] for _ in range(num_first_inst_trades_to_analyze)]
            all_abs_md0_for_corr, all_abs_first_trade_imp_for_corr = [],[]
            gap_closed_to_min_all_events, crossed_zero_post_event_flags = [], []
            time_to_first_reversion_bars = []

            for i in ev_idx:
                t0_event_bar, p0_event_bar_yes, md0_event_bar = tb_idx_current[i], tb_yes_current_np[i], tb_ma_diff_current_np[i]
                price_at_event_bar = tb_px_current_np[i]
                ma_at_event_bar = tb_ma_current_np[i]

                if p0_event_bar_yes in (0, 1) or pd.isna(md0_event_bar) or pd.isna(ma_at_event_bar) or ma_at_event_bar == 0:
                    continue
                
                if ma_diff_method == "z_score":
                    market_specific_zscore_event_hours.append(t0_event_bar.hour)

                pre_s_idx = t_idx_global.searchsorted(t0_event_bar - current_pre_window_duration)
                pre_e_idx = t_idx_global.searchsorted(t0_event_bar)
                if pre_s_idx == pre_e_idx: continue
                
                pre_lbl_segment, pre_sz_segment = t_lbl_global[pre_s_idx:pre_e_idx], t_sz_global[pre_s_idx:pre_e_idx]
                if pre_sz_segment.sum() == 0: continue

                cnt_share_retail_pre = (pre_lbl_segment == "Retail").sum() / (pre_e_idx - pre_s_idx)
                flow_share_retail_pre = pre_sz_segment[pre_lbl_segment == "Retail"].sum() / pre_sz_segment.sum()

                is_retail_driven = (retail_typ == "count" and cnt_share_retail_pre > retail_thr) or \
                                   (retail_typ == "flow" and flow_share_retail_pre > retail_thr)
                if not is_retail_driven: continue
                ev += 1

                post_s_idx_trades = pre_e_idx 
                post_e_idx_trades = t_idx_global.searchsorted(t0_event_bar + current_post_window_duration)
                if post_s_idx_trades == post_e_idx_trades: continue

                post_lbl_segment_trades = t_lbl_global[post_s_idx_trades:post_e_idx_trades]
                post_sz_segment_trades = t_sz_global[post_s_idx_trades:post_e_idx_trades]
                post_yes_segment_trades = t_yes_global[post_s_idx_trades:post_e_idx_trades]
                post_time_segment_trades = t_idx_global[post_s_idx_trades:post_e_idx_trades]
                
                inst_mask_post_trades = (post_lbl_segment_trades == "Institutional")
                if not inst_mask_post_trades.any() or post_sz_segment_trades.sum() == 0: continue
                at += 1
                
                inst_share_in_post = post_sz_segment_trades[inst_mask_post_trades].sum() / post_sz_segment_trades.sum()
                inst_all_shares_post_event.append(inst_share_in_post)

                inst_trades_yes_prices_in_post = post_yes_segment_trades[inst_mask_post_trades]
                current_event_first_N_impacts = [np.nan] * num_first_inst_trades_to_analyze
                current_event_first_N_correct_dir = [False] * num_first_inst_trades_to_analyze

                for k_trade_idx in range(min(num_first_inst_trades_to_analyze, len(inst_trades_yes_prices_in_post))):
                    kth_inst_trade_yes_price = inst_trades_yes_prices_in_post[k_trade_idx]
                    kth_impact_delta_p = kth_inst_trade_yes_price - p0_event_bar_yes
                    current_event_first_N_impacts[k_trade_idx] = kth_impact_delta_p
                    
                    is_correct_dir = False
                    if md0_event_bar > 1e-9 and kth_impact_delta_p < 0: is_correct_dir = True
                    elif md0_event_bar < -1e-9 and kth_impact_delta_p > 0: is_correct_dir = True
                    current_event_first_N_correct_dir[k_trade_idx] = is_correct_dir
                
                if not pd.isna(current_event_first_N_impacts[0]):
                    all_abs_first_trade_imp_for_corr.append(np.abs(current_event_first_N_impacts[0]))
                    all_abs_md0_for_corr.append(np.abs(md0_event_bar))

                tb_post_s_idx_bars = tb_idx_current.searchsorted(t0_event_bar, side="right")
                tb_post_e_idx_bars = tb_idx_current.searchsorted(t0_event_bar + current_post_window_duration, side="right")

                closed_pct_final, closed_pct_to_min_in_window, price_crossed_zero_ma_diff = np.nan, np.nan, False
                current_time_to_reversion_bars = np.nan

                if tb_post_s_idx_bars < tb_post_e_idx_bars:
                    post_window_ma_diffs_bars = tb_ma_diff_current_np[tb_post_s_idx_bars : tb_post_e_idx_bars]
                    
                    ma_diff_at_end_of_window = post_window_ma_diffs_bars[-1]
                    if abs(md0_event_bar) > 1e-9:
                        closed_pct_final = 1 - (abs(ma_diff_at_end_of_window) / abs(md0_event_bar))
                        min_abs_ma_diff_in_post_window = np.min(np.abs(post_window_ma_diffs_bars))
                        closed_pct_to_min_in_window = (abs(md0_event_bar) - min_abs_ma_diff_in_post_window) / abs(md0_event_bar)

                    if np.sign(md0_event_bar) != 0:
                        crossed_indices = np.where( (np.sign(post_window_ma_diffs_bars) != np.sign(md0_event_bar)) & (np.sign(post_window_ma_diffs_bars) != 0) | (post_window_ma_diffs_bars == 0) )[0]
                        if crossed_indices.size > 0:
                            price_crossed_zero_ma_diff = True
                            current_time_to_reversion_bars = crossed_indices[0] + 1
                
                gap_closed_all_events.append(closed_pct_final)
                gap_closed_to_min_all_events.append(closed_pct_to_min_in_window)
                crossed_zero_post_event_flags.append(price_crossed_zero_ma_diff)
                if not pd.isna(current_time_to_reversion_bars):
                    time_to_first_reversion_bars.append(current_time_to_reversion_bars)

                initial_ma_diff_pct_for_success_metric = md0_event_bar
                if ma_diff_method == "z_score" and ma_at_event_bar != 0:
                     initial_ma_diff_pct_for_success_metric = (price_at_event_bar - ma_at_event_bar) / ma_at_event_bar
                elif ma_diff_method == "z_score" and ma_at_event_bar == 0:
                     initial_ma_diff_pct_for_success_metric = np.nan

                is_successful_correction = False
                if not pd.isna(closed_pct_final) and not pd.isna(initial_ma_diff_pct_for_success_metric) and abs(initial_ma_diff_pct_for_success_metric) > 1e-9:
                    if closed_pct_final >= (reversion_tol_default / abs(initial_ma_diff_pct_for_success_metric)):
                        is_successful_correction = True
                
                if is_successful_correction:
                    su += 1
                    inst_succ_shares.append(inst_share_in_post)
                    for k_trade_idx in range(num_first_inst_trades_to_analyze):
                        first_N_inst_trade_impacts_succ[k_trade_idx].append(current_event_first_N_impacts[k_trade_idx])
                        first_N_inst_trade_correct_dir_succ[k_trade_idx].append(current_event_first_N_correct_dir[k_trade_idx])
                else:
                    inst_fail_shares.append(inst_share_in_post)
                    for k_trade_idx in range(num_first_inst_trades_to_analyze):
                        first_N_inst_trade_impacts_fail[k_trade_idx].append(current_event_first_N_impacts[k_trade_idx])
                        first_N_inst_trade_correct_dir_fail[k_trade_idx].append(current_event_first_N_correct_dir[k_trade_idx])

                if events_detected_for_detailed_log < max_detailed_events_to_log_per_config:
                    log_detail_post_lbl = post_lbl_segment_trades
                    log_detail_post_sz = post_sz_segment_trades
                    log_detail_cum_retail_flow = np.cumsum(np.where(log_detail_post_lbl == "Retail", log_detail_post_sz, 0))
                    log_detail_cum_inst_flow = np.cumsum(np.where(log_detail_post_lbl == "Institutional", log_detail_post_sz, 0))
                    
                    first_N_inst_trades_details_for_log = []
                    inst_trade_indices_in_post = np.where(inst_mask_post_trades)[0]
                    for k_log_idx in range(min(num_first_inst_trades_to_analyze, len(inst_trade_indices_in_post))):
                        actual_trade_idx_in_segment = inst_trade_indices_in_post[k_log_idx]
                        first_N_inst_trades_details_for_log.append({
                            f"inst_trade_{k_log_idx+1}_yes_price": post_yes_segment_trades[actual_trade_idx_in_segment],
                            f"inst_trade_{k_log_idx+1}_size": post_sz_segment_trades[actual_trade_idx_in_segment],
                            f"inst_trade_{k_log_idx+1}_time_offset_ms": int((post_time_segment_trades[actual_trade_idx_in_segment] - t0_event_bar).total_seconds() * 1000)
                        })
                    
                    event_details_log_entry = {
                        "market": market_name,
                        "exp_ma_window": ma_win, "exp_ma_diff_method": ma_diff_method,
                        "exp_pre_window_duration_str": str(current_pre_window_duration),
                        "exp_post_window_duration_str": str(current_post_window_duration),
                        "retail_filter_type": retail_typ, "retail_filter_threshold": retail_thr,
                        "event_time_t0": t0_event_bar, "price_at_t0_yes": p0_event_bar_yes,
                        "ma_at_t0_value": ma_at_event_bar, "ma_diff_at_t0_value": md0_event_bar,
                        "closed_pct_final": closed_pct_final, "closed_pct_to_min_in_window": closed_pct_to_min_in_window,
                        "is_successful_correction": is_successful_correction,
                        "inst_share_in_post": inst_share_in_post,
                        "price_series_tb_yes_post": tb_yes_current_np[tb_post_s_idx_bars:tb_post_e_idx_bars].tolist() if tb_post_s_idx_bars < tb_post_e_idx_bars else [],
                        "ma_series_tb_post": tb_ma_current_np[tb_post_s_idx_bars:tb_post_e_idx_bars].tolist() if tb_post_s_idx_bars < tb_post_e_idx_bars else [],
                        "cum_retail_flow_post_trades": log_detail_cum_retail_flow.tolist(),
                        "cum_inst_flow_post_trades": log_detail_cum_inst_flow.tolist(),
                        **{k:v for d in first_N_inst_trades_details_for_log for k,v in d.items()}
                    }
                    market_specific_detailed_events.append(event_details_log_entry)
                    events_detected_for_detailed_log += 1
            
            record_entry_base = {
                "market": market_name,
                "exp_ma_window": ma_win, "exp_ma_diff_method": ma_diff_method,
                "exp_ma_detection_threshold": round(current_ma_detection_threshold,3),
                "exp_pre_window_duration_str": str(current_pre_window_duration),
                "exp_post_window_duration_str": str(current_post_window_duration),
                "retail_filter_type": retail_typ, "retail_filter_threshold": retail_thr,
                "events_retail_driven": ev, "events_inst_attempt": at, "events_successful_correction": su,
            }

            if ev == 0:
                nan_metrics = {k: np.nan for k in [
                    "attempt_rate", "success_rate", "mean_inst_share_succ", "mean_inst_share_fail", 
                    "mwu_p_inst_share", "avg_gap_closed_final", "avg_gap_closed_to_min", 
                    "avg_crossed_zero_flag", "avg_time_to_reversion_min",
                    "corr_inst_share_vs_gap_closed", "corr_inst_share_vs_gap_closed_p",
                    "corr_abs_first_trade_imp_vs_abs_md0", "corr_abs_first_trade_imp_vs_abs_md0_p"
                ]}
                for k_trade_idx in range(num_first_inst_trades_to_analyze):
                    nan_metrics[f"mean_first_{k_trade_idx+1}_imp_succ"] = np.nan
                    nan_metrics[f"mean_first_{k_trade_idx+1}_imp_fail"] = np.nan
                    nan_metrics[f"mean_first_{k_trade_idx+1}_correct_dir_succ"] = np.nan
                    nan_metrics[f"mean_first_{k_trade_idx+1}_correct_dir_fail"] = np.nan
                
                market_specific_records.append({**record_entry_base, **nan_metrics, "corr_inst_share_nan_reason": "No events"})
                line_entry = f"\nConfig: MA_win={ma_win}, MA_diff='{ma_diff_method}'(thr={current_ma_detection_threshold:.2f}), PreW={str(current_pre_window_duration)}, PostW={str(current_post_window_duration)}, Retail='{retail_typ}>{retail_thr}'\nEvents: {ev} (No retail-driven events)"
                market_specific_lines.append(line_entry)
                continue

            attempt_rate = at / ev if ev else np.nan
            success_rate = su / at if at else np.nan
            
            succ_arr_np, fail_arr_np = np.array(inst_succ_shares), np.array(inst_fail_shares)
            mwu_p = mannwhitneyu(succ_arr_np, fail_arr_np, nan_policy='omit', alternative='two-sided').pvalue if succ_arr_np.size > 0 and fail_arr_np.size > 0 else np.nan

            corr_inst_gap, corr_inst_gap_p, reason_nan_corr_inst_gap = np.nan, np.nan, "N/A"
            valid_idx_corr1 = ~np.isnan(inst_all_shares_post_event) & ~np.isnan(gap_closed_all_events)
            series1_corr1, series2_corr1 = np.array(inst_all_shares_post_event)[valid_idx_corr1], np.array(gap_closed_all_events)[valid_idx_corr1]
            if len(series1_corr1) >= min_events_for_correlation:
                if np.std(series1_corr1) > 1e-9 and np.std(series2_corr1) > 1e-9: corr_inst_gap, corr_inst_gap_p = pearsonr(series1_corr1, series2_corr1)
                else: reason_nan_corr_inst_gap = "Zero variance in one/both series"
            else: reason_nan_corr_inst_gap = f"<{min_events_for_correlation} valid pairs"

            corr_fimp_md0, corr_fimp_md0_p = np.nan, np.nan
            valid_idx_corr2 = ~np.isnan(all_abs_first_trade_imp_for_corr) & ~np.isnan(all_abs_md0_for_corr)
            series1_corr2, series2_corr2 = np.array(all_abs_md0_for_corr)[valid_idx_corr2], np.array(all_abs_first_trade_imp_for_corr)[valid_idx_corr2]
            if len(series1_corr2) >= min_events_for_correlation:
                 if np.std(series1_corr2) > 1e-9 and np.std(series2_corr2) > 1e-9: corr_fimp_md0, corr_fimp_md0_p = pearsonr(series1_corr2, series2_corr2)
            
            current_metrics = {
                "attempt_rate": attempt_rate, "success_rate": success_rate,
                "mean_inst_share_succ": np.nanmean(succ_arr_np) if succ_arr_np.size > 0 else np.nan,
                "mean_inst_share_fail": np.nanmean(fail_arr_np) if fail_arr_np.size > 0 else np.nan,
                "mwu_p_inst_share": mwu_p,
                "avg_gap_closed_final": np.nanmean(gap_closed_all_events) if len(gap_closed_all_events) > 0 else np.nan,
                "avg_gap_closed_to_min": np.nanmean(gap_closed_to_min_all_events) if len(gap_closed_to_min_all_events) > 0 else np.nan,
                "avg_crossed_zero_flag": np.nanmean(crossed_zero_post_event_flags) if len(crossed_zero_post_event_flags) > 0 else np.nan,
                "avg_time_to_reversion_min": np.nanmean(time_to_first_reversion_bars) if len(time_to_first_reversion_bars) > 0 else np.nan,
                "corr_inst_share_vs_gap_closed": corr_inst_gap, "corr_inst_share_vs_gap_closed_p": corr_inst_gap_p,
                "corr_inst_share_nan_reason": reason_nan_corr_inst_gap if np.isnan(corr_inst_gap) else "",
                "corr_abs_first_trade_imp_vs_abs_md0": corr_fimp_md0, "corr_abs_first_trade_imp_vs_abs_md0_p": corr_fimp_md0_p,
            }
            for k_trd in range(num_first_inst_trades_to_analyze):
                current_metrics[f"mean_first_{k_trd+1}_imp_succ"] = np.nanmean([imp for imp in first_N_inst_trade_impacts_succ[k_trd] if not pd.isna(imp)]) if len(first_N_inst_trade_impacts_succ[k_trd]) > 0 else np.nan
                current_metrics[f"mean_first_{k_trd+1}_imp_fail"] = np.nanmean([imp for imp in first_N_inst_trade_impacts_fail[k_trd] if not pd.isna(imp)]) if len(first_N_inst_trade_impacts_fail[k_trd]) > 0 else np.nan
                current_metrics[f"mean_first_{k_trd+1}_correct_dir_succ"] = np.nanmean([cd for cd in first_N_inst_trade_correct_dir_succ[k_trd] if isinstance(cd, bool)]) if len(first_N_inst_trade_correct_dir_succ[k_trd]) > 0 else np.nan
                current_metrics[f"mean_first_{k_trd+1}_correct_dir_fail"] = np.nanmean([cd for cd in first_N_inst_trade_correct_dir_fail[k_trd] if isinstance(cd, bool)]) if len(first_N_inst_trade_correct_dir_fail[k_trd]) > 0 else np.nan
            
            market_specific_records.append({**record_entry_base, **current_metrics})
            
            line_parts = [
                f"\nConfig: MA_win={ma_win}, MA_diff='{ma_diff_method}'(thr={current_ma_detection_threshold:.2f}), PreW={str(current_pre_window_duration)}, PostW={str(current_post_window_duration)}, Retail='{retail_typ}>{retail_thr}'",
                f"Events: {ev}, Inst. Attempts: {at} (rate: {current_metrics['attempt_rate']:.3%}), Successes: {su} (rate: {current_metrics['success_rate']:.3%})",
                f"Inst. Share (Succ|Fail): {current_metrics['mean_inst_share_succ']:.3%} | {current_metrics['mean_inst_share_fail']:.3%} (MWU p: {current_metrics['mwu_p_inst_share']:.2e})",
                f"Gap Closed (Final|ToMin): {current_metrics['avg_gap_closed_final']:.3f} | {current_metrics['avg_gap_closed_to_min']:.3f}. Crossed MA Zero: {current_metrics['avg_crossed_zero_flag']:.1%}. Avg Reversion Time: {current_metrics['avg_time_to_reversion_min']:.1f} min",
                f"Corr(InstShare,GapFinal): {current_metrics['corr_inst_share_vs_gap_closed']:.3f} (p={current_metrics['corr_inst_share_vs_gap_closed_p']:.2e}) {current_metrics['corr_inst_share_nan_reason']}",
            ]
            for k_trd in range(num_first_inst_trades_to_analyze):
                line_parts.append(
                    f"Mean {k_trd+1}st Inst ΔP (Succ|Fail): {current_metrics[f'mean_first_{k_trd+1}_imp_succ']:.4f} ({current_metrics[f'mean_first_{k_trd+1}_correct_dir_succ']:.1%}) | {current_metrics[f'mean_first_{k_trd+1}_imp_fail']:.4f} ({current_metrics[f'mean_first_{k_trd+1}_correct_dir_fail']:.1%})"
                )
            line_parts.append(f"Corr(|1st Imp|,|Initial MD0|): {current_metrics['corr_abs_first_trade_imp_vs_abs_md0']:.3f} (p={current_metrics['corr_abs_first_trade_imp_vs_abs_md0_p']:.2e})")
            market_specific_lines.append("\n".join(line_parts))

    if market_specific_zscore_event_hours:
        hour_counts = Counter(market_specific_zscore_event_hours)
        market_specific_lines.append(f"\nZ-Score Event Frequency by Hour for market {market_name}:")
        for hour in sorted(hour_counts.keys()):
            market_specific_lines.append(f"  Hour {hour:02d}: {hour_counts[hour]} events")

    pd.DataFrame(market_specific_records).to_csv(table_path, index=False)
    print(f"\nMetrics table for {market_name} saved to {table_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"RETAIL MISPRICING ➜ INSTITUTIONAL CORRECTION : FURTHER EXPLORATION STUDY\n")
        f.write(f"Market: {market_name}\n")
        f.write(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Common fixed parameters for this run:\n")
        f.write(f"  Reversion Tolerance for Success (reversion_tol_default): {reversion_tol_default}\n")
        f.write(f"  Retail Count Share Threshold (cnt_thr_default): {cnt_thr_default}\n")
        f.write(f"  Retail Flow Share Thresholds (flow_thrs_default): {flow_thrs_default}\n")
        f.write(f"  Z-Score MA Diff Rolling Window: {z_score_rolling_window}\n\n")
        for ln in market_specific_lines:
            f.write(ln)
    print(f"Text report for {market_name} saved to {report_path}")

    if market_specific_detailed_events:
        pd.DataFrame(market_specific_detailed_events).to_csv(detailed_events_log_path, index=False)
        print(f"Detailed event log for {market_name} saved to {detailed_events_log_path}")

    all_market_run_records.extend(market_specific_records)
    all_market_run_lines.extend(market_specific_lines)
    all_market_detailed_events_data.extend(market_specific_detailed_events)

if all_market_run_records:
    combined_table_path = base_out_dir / "ALL_MARKETS_combined_metrics.csv"
    pd.DataFrame(all_market_run_records).to_csv(combined_table_path, index=False)
    print(f"\nCombined metrics for ALL markets saved to {combined_table_path}")

if all_market_detailed_events_data:
    combined_detailed_log_path = base_out_dir / "ALL_MARKETS_combined_detailed_events.csv"
    pd.DataFrame(all_market_detailed_events_data).to_csv(combined_detailed_log_path, index=False)
    print(f"Combined detailed event log for ALL markets saved to {combined_detailed_log_path}")

if all_market_run_lines:
    combined_report_path = base_out_dir / "ALL_MARKETS_combined_report.txt"
    with open(combined_report_path, "w", encoding="utf-8") as f:
        f.write("RETAIL MISPRICING ➜ INSTITUTIONAL CORRECTION : COMBINED FURTHER EXPLORATION STUDY\n")
        f.write(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Common fixed parameters for all runs:\n")
        f.write(f"  Reversion Tolerance for Success (reversion_tol_default): {reversion_tol_default}\n")
        f.write(f"  Retail Count Share Threshold (cnt_thr_default): {cnt_thr_default}\n")
        f.write(f"  Retail Flow Share Thresholds (flow_thrs_default): {flow_thrs_default}\n")
        f.write(f"  Z-Score MA Diff Rolling Window: {z_score_rolling_window}\n\n")
        for ln in all_market_run_lines:
            f.write(ln)
    print(f"Combined text report for ALL markets saved to {combined_report_path}")

