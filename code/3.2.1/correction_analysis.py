"""
analyzes trade and timebar data to identify retail-driven mispricings and subsequent institutional corrections across various parameter configurations.
"""
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm
from scipy.stats import mannwhitneyu, pearsonr

out_dir = Path("./")
out_dir.mkdir(exist_ok=True)
report_path = out_dir / "institution_correction_extended_report.txt"
table_path = out_dir / "institution_correction_extended_metrics.csv"
detailed_events_log_path = out_dir / "detailed_event_examples_log.csv"

trades_master = pd.read_parquet("clean_trades.parquet")
trades_master["time"] = pd.to_datetime(trades_master["time"])
trades_master = trades_master.sort_values("time").reset_index(drop=True)

tb_master = pd.read_parquet("timebars.parquet")
tb_master["time"] = pd.to_datetime(tb_master["time"])
tb_master = tb_master.sort_values("time").reset_index(drop=True)

t_idx_global = pd.DatetimeIndex(trades_master["time"])
t_sz_global = trades_master["size"].values
t_lbl_global = trades_master["label"].values
t_px_global = trades_master["price"].values
t_yes_global = np.where(t_px_global <= 50, (100 - t_px_global) / 100, t_px_global / 100)

baseline_inst_share = trades_master.loc[trades_master["label"] == "Institutional", "size"].sum() / trades_master["size"].sum()

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
min_events_for_correlation = 5
max_detailed_events_to_log_per_config = 3

all_run_records = []
all_run_lines = []
detailed_events_data_all_runs = []

experiment_configs = []
for ma_win_opt in ma_window_options:
    for ma_diff_method_opt in ma_diff_calculation_methods:
        for pre_window_opt in pre_window_duration_options:
            for post_window_opt in post_window_duration_options:
                experiment_configs.append({
                    "ma_window": ma_win_opt,
                    "ma_diff_method": ma_diff_method_opt,
                    "pre_window_duration": pre_window_opt,
                    "post_window_duration": post_window_opt
                })

for exp_config in tqdm(experiment_configs, desc="Overall Experiments"):
    ma_win = exp_config["ma_window"]
    ma_diff_method = exp_config["ma_diff_method"]
    current_pre_window_duration = exp_config["pre_window_duration"]
    current_post_window_duration = exp_config["post_window_duration"]

    tb = tb_master.copy()

    ma_col_name = f"ma{ma_win}"
    tb[ma_col_name] = tb["price"].rolling(ma_win, min_periods=max(1, ma_win//2)).mean().bfill().ffill()

    if ma_diff_method == "percentage":
        tb["ma_diff_current"] = ((tb["price"] - tb[ma_col_name]) / tb[ma_col_name].replace(0, np.nan)).fillna(0)
        current_ma_detection_threshold = ma_thr_pct_default
    elif ma_diff_method == "z_score":
        ma_diff_raw_pct = ((tb["price"] - tb[ma_col_name]) / tb[ma_col_name].replace(0, np.nan)).fillna(0)
        ma_diff_rolling_std = ma_diff_raw_pct.rolling(window=z_score_rolling_window, min_periods=max(1, z_score_rolling_window//2)).std().bfill().ffill()
        tb["ma_diff_current"] = (ma_diff_raw_pct / ma_diff_rolling_std.replace(0, np.nan)).fillna(0)
        current_ma_detection_threshold = ma_thr_z_default
    else:
        raise ValueError(f"Unknown ma_diff_method: {ma_diff_method}")

    tb_idx_current = pd.DatetimeIndex(tb["time"])
    tb_px_current_np = tb["price"].values
    tb_yes_current_np = np.where(tb_px_current_np <= 50, (100 - tb_px_current_np) / 100, tb_px_current_np / 100)
    tb_ma_diff_current_np = tb["ma_diff_current"].values
    tb_ma_current_np = tb[ma_col_name].values

    retail_definition_configs = [("count", cnt_thr_default)] + [("flow", thr) for thr in flow_thrs_default]
    
    for retail_typ, retail_thr in retail_definition_configs:
        events_detected_for_detailed_log = 0

        ev_idx = np.where(np.abs(tb_ma_diff_current_np) > current_ma_detection_threshold)[0]
        
        ev = 0
        at = 0
        su = 0

        inst_succ_shares, inst_fail_shares = [], []
        inst_all_shares_post_event, gap_closed_all_events = [], []
        
        first_imp_succ, first_imp_fail = [], []
        first_imp_succ_correct_dir_flags, first_imp_fail_correct_dir_flags = [], []
        
        all_abs_first_imp_for_corr, all_abs_md0_for_corr = [], []

        gap_closed_to_min_all_events = []
        crossed_zero_post_event_flags = []

        for i in ev_idx:
            t0_event_bar, p0_event_bar_yes, md0_event_bar = tb_idx_current[i], tb_yes_current_np[i], tb_ma_diff_current_np[i]
            price_at_event_bar = tb_px_current_np[i]
            ma_at_event_bar = tb_ma_current_np[i]

            if p0_event_bar_yes in (0, 1) or pd.isna(md0_event_bar) or pd.isna(ma_at_event_bar) or ma_at_event_bar == 0:
                continue

            pre_s_idx = t_idx_global.searchsorted(t0_event_bar - current_pre_window_duration)
            pre_e_idx = t_idx_global.searchsorted(t0_event_bar)

            if pre_s_idx == pre_e_idx:
                continue
            
            pre_lbl_segment, pre_sz_segment = t_lbl_global[pre_s_idx:pre_e_idx], t_sz_global[pre_s_idx:pre_e_idx]
            
            if pre_sz_segment.sum() == 0:
                continue

            cnt_share_retail_pre = (pre_lbl_segment == "Retail").sum() / (pre_e_idx - pre_s_idx)
            flow_share_retail_pre = pre_sz_segment[pre_lbl_segment == "Retail"].sum() / pre_sz_segment.sum()

            is_retail_driven = False
            if retail_typ == "count" and cnt_share_retail_pre > retail_thr:
                is_retail_driven = True
            elif retail_typ == "flow" and flow_share_retail_pre > retail_thr:
                is_retail_driven = True
            
            if not is_retail_driven:
                continue
            ev += 1

            post_s_idx = pre_e_idx
            post_e_idx = t_idx_global.searchsorted(t0_event_bar + current_post_window_duration)
            
            if post_s_idx == post_e_idx:
                continue

            post_lbl_segment = t_lbl_global[post_s_idx:post_e_idx]
            post_sz_segment = t_sz_global[post_s_idx:post_e_idx]
            post_yes_segment = t_yes_global[post_s_idx:post_e_idx]
            
            inst_mask_post = (post_lbl_segment == "Institutional")
            if not inst_mask_post.any() or post_sz_segment.sum() == 0:
                continue
            at += 1
            
            inst_share_in_post = post_sz_segment[inst_mask_post].sum() / post_sz_segment.sum()
            inst_all_shares_post_event.append(inst_share_in_post)

            first_inst_trade_yes_price_post = post_yes_segment[inst_mask_post][0]
            first_impact_delta_p = first_inst_trade_yes_price_post - p0_event_bar_yes
            all_abs_first_imp_for_corr.append(np.abs(first_impact_delta_p))
            all_abs_md0_for_corr.append(np.abs(md0_event_bar))

            first_imp_moves_correct_direction = False
            if md0_event_bar > 1e-9 and first_impact_delta_p < 0:
                first_imp_moves_correct_direction = True
            elif md0_event_bar < -1e-9 and first_impact_delta_p > 0:
                first_imp_moves_correct_direction = True

            tb_post_s_idx = tb_idx_current.searchsorted(t0_event_bar, side="right")
            tb_post_e_idx = tb_idx_current.searchsorted(t0_event_bar + current_post_window_duration, side="right")

            closed_pct_final = np.nan
            closed_pct_to_min_in_window = np.nan
            price_crossed_zero_ma_diff = False

            if tb_post_s_idx < tb_post_e_idx :
                ma_diff_at_end_of_window = tb_ma_diff_current_np[tb_post_e_idx - 1]
                
                if abs(md0_event_bar) > 1e-9:
                    gap_raw_final = abs(ma_diff_at_end_of_window) / abs(md0_event_bar)
                    closed_pct_final = 1 - gap_raw_final
                    
                    min_abs_ma_diff_in_post_window = np.min(np.abs(tb_ma_diff_current_np[tb_post_s_idx : tb_post_e_idx]))
                    closed_pct_to_min_in_window = (abs(md0_event_bar) - min_abs_ma_diff_in_post_window) / abs(md0_event_bar)

                if len(tb_ma_diff_current_np[tb_post_s_idx : tb_post_e_idx]) > 0:
                    post_window_ma_diffs = tb_ma_diff_current_np[tb_post_s_idx : tb_post_e_idx]
                    if np.sign(md0_event_bar) != 0 and \
                       (np.any(np.sign(post_window_ma_diffs) != np.sign(md0_event_bar)) or np.any(post_window_ma_diffs == 0)):
                        price_crossed_zero_ma_diff = True
            
            gap_closed_all_events.append(closed_pct_final)
            gap_closed_to_min_all_events.append(closed_pct_to_min_in_window)
            crossed_zero_post_event_flags.append(price_crossed_zero_ma_diff)
            
            initial_ma_diff_pct_for_success_metric = md0_event_bar
            if ma_diff_method == "z_score":
                if ma_at_event_bar != 0:
                     initial_ma_diff_pct_for_success_metric = (price_at_event_bar - ma_at_event_bar) / ma_at_event_bar
                else:
                     initial_ma_diff_pct_for_success_metric = np.nan

            is_successful_correction = False
            if not pd.isna(closed_pct_final) and not pd.isna(initial_ma_diff_pct_for_success_metric) and abs(initial_ma_diff_pct_for_success_metric) > 1e-9:
                if closed_pct_final >= (reversion_tol_default / abs(initial_ma_diff_pct_for_success_metric)):
                    is_successful_correction = True
            
            if is_successful_correction:
                su += 1
                inst_succ_shares.append(inst_share_in_post)
                first_imp_succ.append(first_impact_delta_p)
                first_imp_succ_correct_dir_flags.append(first_imp_moves_correct_direction)
            else:
                inst_fail_shares.append(inst_share_in_post)
                first_imp_fail.append(first_impact_delta_p)
                first_imp_fail_correct_dir_flags.append(first_imp_moves_correct_direction)

            if events_detected_for_detailed_log < max_detailed_events_to_log_per_config:
                event_details_log_entry = {
                    "exp_ma_window": ma_win, "exp_ma_diff_method": ma_diff_method,
                    "exp_pre_window_duration_str": str(current_pre_window_duration),
                    "exp_post_window_duration_str": str(current_post_window_duration),
                    "retail_filter_type": retail_typ, "retail_filter_threshold": retail_thr,
                    "event_time_t0": t0_event_bar,
                    "price_at_t0_yes": p0_event_bar_yes,
                    "ma_at_t0_value": ma_at_event_bar,
                    "ma_diff_at_t0_value": md0_event_bar,
                    "initial_ma_diff_pct_for_success_metric": initial_ma_diff_pct_for_success_metric,
                    "end_of_window_time": tb_idx_current[tb_post_e_idx - 1] if tb_post_s_idx < tb_post_e_idx else None,
                    "ma_diff_at_end_value": tb_ma_diff_current_np[tb_post_e_idx - 1] if tb_post_s_idx < tb_post_e_idx else None,
                    "closed_pct_final": closed_pct_final,
                    "closed_pct_to_min_in_window": closed_pct_to_min_in_window,
                    "price_crossed_zero_ma_diff_flag": price_crossed_zero_ma_diff,
                    "is_successful_correction": is_successful_correction,
                    "inst_share_in_post": inst_share_in_post,
                    "first_inst_trade_yes_price_post": first_inst_trade_yes_price_post,
                    "first_impact_delta_p": first_impact_delta_p,
                    "first_imp_moves_correct_direction": first_imp_moves_correct_direction,
                }
                detailed_events_data_all_runs.append(event_details_log_entry)
                events_detected_for_detailed_log += 1
        
        if ev == 0:
            record_entry = {
                "exp_ma_window": ma_win, "exp_ma_diff_method": ma_diff_method,
                "exp_ma_detection_threshold": current_ma_detection_threshold,
                "exp_pre_window_duration_str": str(current_pre_window_duration),
                "exp_post_window_duration_str": str(current_post_window_duration),
                "retail_filter_type": retail_typ, "retail_filter_threshold": retail_thr,
                "events_retail_driven": ev, "events_inst_attempt": at, "events_successful_correction": su,
                "attempt_rate": np.nan, "success_rate": np.nan,
                "mean_inst_share_succ": np.nan, "mean_inst_share_fail": np.nan, "mwu_p_inst_share": np.nan,
                "avg_gap_closed_final": np.nan, "avg_gap_closed_to_min": np.nan, "avg_crossed_zero_flag": np.nan,
                "corr_inst_share_vs_gap_closed": np.nan, "corr_inst_share_vs_gap_closed_p": np.nan,
                "corr_inst_share_nan_reason": "No events",
                "mean_first_imp_succ": np.nan, "mean_first_imp_fail": np.nan,
                "mean_first_imp_correct_dir_succ": np.nan, "mean_first_imp_correct_dir_fail": np.nan,
                "corr_abs_first_imp_vs_abs_md0": np.nan, "corr_abs_first_imp_vs_abs_md0_p": np.nan,
            }
            all_run_records.append(record_entry)
            line_entry = f"""
----------------------------------------
Config: MA_win={ma_win}, MA_diff_method='{ma_diff_method}' (thr={current_ma_detection_threshold:.2f}), PreWindow={str(current_pre_window_duration)}, PostWindow={str(current_post_window_duration)}
Retail Pre-filter       : {retail_typ}>{retail_thr}
Events detected         : {ev} (No retail-driven events matching criteria)
"""
            all_run_lines.append(line_entry)
            continue

        succ_arr_np, fail_arr_np = np.array(inst_succ_shares), np.array(inst_fail_shares)
        mwu_p = mannwhitneyu(succ_arr_np, fail_arr_np, nan_policy='omit').pvalue if succ_arr_np.size > 0 and fail_arr_np.size > 0 else np.nan

        corr_inst_gap, corr_inst_gap_p = np.nan, np.nan
        reason_for_nan_corr_inst_gap = "N/A"
        valid_corr_indices = ~np.isnan(inst_all_shares_post_event) & ~np.isnan(gap_closed_all_events)
        inst_all_corr_valid = np.array(inst_all_shares_post_event)[valid_corr_indices]
        gap_closed_all_corr_valid = np.array(gap_closed_all_events)[valid_corr_indices]

        if len(inst_all_corr_valid) >= min_events_for_correlation:
            std_inst_all = np.std(inst_all_corr_valid)
            std_gap_closed = np.std(gap_closed_all_corr_valid)
            if std_inst_all > 1e-9 and std_gap_closed > 1e-9:
                corr_inst_gap, corr_inst_gap_p = pearsonr(inst_all_corr_valid, gap_closed_all_corr_valid)
            elif std_inst_all <= 1e-9 and std_gap_closed <= 1e-9: reason_for_nan_corr_inst_gap = "Zero variance in both series"
            elif std_inst_all <= 1e-9: reason_for_nan_corr_inst_gap = "Zero variance in inst_share"
            else: reason_for_nan_corr_inst_gap = "Zero variance in gap_closed"
        else:
            reason_for_nan_corr_inst_gap = f"Too few data points for correlation (<{min_events_for_correlation} valid pairs)"
        
        corr_fimp_md0, corr_fimp_md0_p = np.nan, np.nan
        valid_fimp_corr_indices = ~np.isnan(all_abs_first_imp_for_corr) & ~np.isnan(all_abs_md0_for_corr)
        abs_fimp_valid = np.array(all_abs_first_imp_for_corr)[valid_fimp_corr_indices]
        abs_md0_valid = np.array(all_abs_md0_for_corr)[valid_fimp_corr_indices]

        if len(abs_fimp_valid) >= min_events_for_correlation:
            std_fimp = np.std(abs_fimp_valid)
            std_md0 = np.std(abs_md0_valid)
            if std_fimp > 1e-9 and std_md0 > 1e-9:
                corr_fimp_md0, corr_fimp_md0_p = pearsonr(abs_md0_valid, abs_fimp_valid)

        record_entry = {
            "exp_ma_window": ma_win, "exp_ma_diff_method": ma_diff_method,
            "exp_ma_detection_threshold": current_ma_detection_threshold,
            "exp_pre_window_duration_str": str(current_pre_window_duration),
            "exp_post_window_duration_str": str(current_post_window_duration),
            "retail_filter_type": retail_typ, "retail_filter_threshold": retail_thr,
            "events_retail_driven": ev,
            "events_inst_attempt": at,
            "events_successful_correction": su,
            "attempt_rate": at / ev if ev else np.nan,
            "success_rate": su / at if at else np.nan,
            "mean_inst_share_succ": np.nanmean(succ_arr_np) if succ_arr_np.size > 0 else np.nan,
            "mean_inst_share_fail": np.nanmean(fail_arr_np) if fail_arr_np.size > 0 else np.nan,
            "mwu_p_inst_share": mwu_p,
            "avg_gap_closed_final": np.nanmean(gap_closed_all_events) if len(gap_closed_all_events) > 0 else np.nan,
            "avg_gap_closed_to_min": np.nanmean(gap_closed_to_min_all_events) if len(gap_closed_to_min_all_events) > 0 else np.nan,
            "avg_crossed_zero_flag": np.nanmean(crossed_zero_post_event_flags) if len(crossed_zero_post_event_flags) > 0 else np.nan,
            "corr_inst_share_vs_gap_closed": corr_inst_gap,
            "corr_inst_share_vs_gap_closed_p": corr_inst_gap_p,
            "corr_inst_share_nan_reason": reason_for_nan_corr_inst_gap if np.isnan(corr_inst_gap) else "",
            "mean_first_imp_succ": np.nanmean(first_imp_succ) if len(first_imp_succ) > 0 else np.nan,
            "mean_first_imp_fail": np.nanmean(first_imp_fail) if len(first_imp_fail) > 0 else np.nan,
            "mean_first_imp_correct_dir_succ": np.nanmean(first_imp_succ_correct_dir_flags) if len(first_imp_succ_correct_dir_flags) > 0 else np.nan,
            "mean_first_imp_correct_dir_fail": np.nanmean(first_imp_fail_correct_dir_flags) if len(first_imp_fail_correct_dir_flags) > 0 else np.nan,
            "corr_abs_first_imp_vs_abs_md0": corr_fimp_md0,
            "corr_abs_first_imp_vs_abs_md0_p": corr_fimp_md0_p,
        }
        all_run_records.append(record_entry)

        line_entry = f"""
----------------------------------------
Config: MA_win={ma_win}, MA_diff_method='{ma_diff_method}' (thr={current_ma_detection_threshold:.2f}), PreWindow={str(current_pre_window_duration)}, PostWindow={str(current_post_window_duration)}
Retail Pre-filter            : {retail_typ}>{retail_thr}
Events retail-driven         : {ev}
Institution attempts         : {at} (rate: {record_entry['attempt_rate']:.3%})
Successful corrections       : {su} (rate: {record_entry['success_rate']:.3%})
Mean inst share (succ)       : {record_entry['mean_inst_share_succ']:.3%}
Mean inst share (fail)       : {record_entry['mean_inst_share_fail']:.3%}
Mann-Whitney U p (inst share): {record_entry['mwu_p_inst_share']:.3e}
Avg Gap Closed (final)       : {record_entry['avg_gap_closed_final']:.3f}
Avg Gap Closed (to min)      : {record_entry['avg_gap_closed_to_min']:.3f}
Avg Crossed MA Zero Flag     : {record_entry['avg_crossed_zero_flag']:.3%}
Corr(InstShare, GapClosed)   : {record_entry['corr_inst_share_vs_gap_closed']:.3f} (p={record_entry['corr_inst_share_vs_gap_closed_p']:.3e}) {record_entry['corr_inst_share_nan_reason']}
Mean 1st Impact Δp (succ)    : {record_entry['mean_first_imp_succ']:.4f} (correct dir: {record_entry['mean_first_imp_correct_dir_succ']:.3%})
Mean 1st Impact Δp (fail)    : {record_entry['mean_first_imp_fail']:.4f} (correct dir: {record_entry['mean_first_imp_correct_dir_fail']:.3%})
Corr(|1st Imp|, |Initial MD0|): {record_entry['corr_abs_first_imp_vs_abs_md0']:.3f} (p={record_entry['corr_abs_first_imp_vs_abs_md0_p']:.3e})
"""
        all_run_lines.append(line_entry)

print("\nSaving all results...")
pd.DataFrame(all_run_records).to_csv(table_path, index=False)
print(f"Metrics table saved to {table_path}")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("RETAIL MISPRICING > INSTITUTIONAL CORRECTION : EXTENDED FULL STUDY\n")
    f.write(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Baseline overall institutional flow-share: {baseline_inst_share:.3%}\n")
    f.write(f"Common fixed parameters:\n")
    f.write(f"  Reversion Tolerance for Success (reversion_tol_default): {reversion_tol_default}\n")
    f.write(f"  Retail Count Share Threshold (cnt_thr_default): {cnt_thr_default}\n")
    f.write(f"  Retail Flow Share Thresholds (flow_thrs_default): {flow_thrs_default}\n")
    f.write(f"  Z-Score MA Diff Rolling Window: {z_score_rolling_window}\n\n")
    
    for ln in all_run_lines:
        f.write(ln)
print(f"Text report saved to {report_path}")

if detailed_events_data_all_runs:
    pd.DataFrame(detailed_events_data_all_runs).to_csv(detailed_events_log_path, index=False)
    print(f"Detailed event examples log saved to {detailed_events_log_path}")
else:
    print("No detailed events were logged (or max_detailed_events_to_log_per_config was 0).")

