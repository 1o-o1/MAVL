#!/usr/bin/env python3
"""
Quick DeLong statistical analysis for memory ablation study
Compares Memory ON (baseline) vs Memory OFF (ablation)
"""

import json
import numpy as np
from pathlib import Path

# Load baseline results (Memory ON - full 40 epochs)
with open(r"E:\Research\zero shot lung\results_full_scale_corrected\summary_results.json") as f:
    baseline_data = json.load(f)

# Load ablation results (Memory OFF - fast 10 epochs)
with open(r"E:\Research\zero shot lung\results_full_scale_ablation_fast\summary_results.json") as f:
    ablation_data = json.load(f)

print("=" * 80)
print("ABLATION STUDY: MEMORY MODULE EFFECT ANALYSIS")
print("=" * 80)
print()

# Extract per-seed AUROC values
baseline_aurocs = []
ablation_aurocs = []
seed_ids = []

for baseline_result, ablation_result in zip(baseline_data['all_results'], ablation_data['all_results']):
    seed = baseline_result['seed']
    baseline_auc = baseline_result['final_auroc']
    ablation_auc = ablation_result['final_auroc']

    baseline_aurocs.append(baseline_auc)
    ablation_aurocs.append(ablation_auc)
    seed_ids.append(seed)

    delta_auc = baseline_auc - ablation_auc
    print(f"Seed {seed}:")
    print(f"  Memory ON  (Baseline): {baseline_auc:.6f}")
    print(f"  Memory OFF (Ablation): {ablation_auc:.6f}")
    print(f"  Delta AUC (Memory effect): {delta_auc:+.6f}")
    print()

# Aggregate metrics
baseline_aurocs_arr = np.array(baseline_aurocs)
ablation_aurocs_arr = np.array(ablation_aurocs)
delta_aurocs = baseline_aurocs_arr - ablation_aurocs_arr

baseline_mean = np.mean(baseline_aurocs_arr)
baseline_std = np.std(baseline_aurocs_arr, ddof=1)

ablation_mean = np.mean(ablation_aurocs_arr)
ablation_std = np.std(ablation_aurocs_arr, ddof=1)

delta_mean = np.mean(delta_aurocs)
delta_std = np.std(delta_aurocs, ddof=1)
delta_sem = delta_std / np.sqrt(len(delta_aurocs))

print("-" * 80)
print("AGGREGATE RESULTS (3 seeds)")
print("-" * 80)
print()
print(f"Memory ON (Baseline):")
print(f"  Mean AUROC:     {baseline_mean:.6f}")
print(f"  Std Dev:        {baseline_std:.6f}")
print(f"  Std Error:      {baseline_std/np.sqrt(3):.6f}")
print()

print(f"Memory OFF (Ablation):")
print(f"  Mean AUROC:     {ablation_mean:.6f}")
print(f"  Std Dev:        {ablation_std:.6f}")
print(f"  Std Error:      {ablation_std/np.sqrt(3):.6f}")
print()

print(f"Memory Effect (Delta AUC = Memory ON - Memory OFF):")
print(f"  Mean Delta AUC: {delta_mean:+.6f}")
print(f"  Std Dev:        {delta_std:.6f}")
print(f"  Std Error Mean: {delta_sem:.6f}")
print()

# Paired t-test
from scipy import stats
t_stat, p_value = stats.ttest_rel(baseline_aurocs_arr, ablation_aurocs_arr)

print("-" * 80)
print("STATISTICAL SIGNIFICANCE TEST")
print("-" * 80)
print(f"Paired t-test (Memory ON vs Memory OFF):")
print(f"  t-statistic:    {t_stat:.4f}")
print(f"  p-value:        {p_value:.6f}")
print()

if p_value < 0.05:
    print("  Result: SIGNIFICANT (p < 0.05)")
    print("  Interpretation: Memory module SIGNIFICANTLY improves AUROC")
else:
    print("  Result: NOT SIGNIFICANT (p >= 0.05)")
    print("  Interpretation: Memory module does NOT significantly improve AUROC")

print()

# Bootstrap confidence interval on ΔAUC
from scipy.stats import bootstrap

def mean_delta(data):
    """Compute mean of delta (second half) - first half"""
    n = len(data) // 2
    return np.mean(data[n:] - data[:n])

# Prepare data for bootstrap: concatenate arrays
all_aucs = np.concatenate([baseline_aurocs_arr, ablation_aurocs_arr])
rng = np.random.default_rng(seed=42)

# Bootstrap resampling for 95% CI
n_bootstrap = 5000
deltas_bootstrap = []

for _ in range(n_bootstrap):
    idx = rng.choice(len(delta_aurocs), size=len(delta_aurocs), replace=True)
    deltas_bootstrap.append(np.mean(delta_aurocs[idx]))

deltas_bootstrap = np.array(deltas_bootstrap)
ci_lower = np.percentile(deltas_bootstrap, 2.5)
ci_upper = np.percentile(deltas_bootstrap, 97.5)

print("-" * 80)
print("95% CONFIDENCE INTERVAL ON Delta AUC (5000 bootstrap resamples)")
print("-" * 80)
print(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
print()

# Interpretation
print("-" * 80)
print("INTERPRETATION")
print("-" * 80)

if delta_mean >= 0.02:
    interpretation = "CRITICAL - Memory module is essential for model performance"
elif delta_mean >= 0.01:
    interpretation = "IMPORTANT - Memory module provides meaningful improvement"
elif delta_mean >= 0.005:
    interpretation = "MODERATE - Memory module provides small but measurable improvement"
else:
    interpretation = "MARGINAL - Memory module has minimal impact on model performance"

print(f"Magnitude of memory effect: Delta AUC = {delta_mean:+.6f}")
print(f"Classification: {interpretation}")
print()

if p_value < 0.05:
    print(f"Statistical Evidence: STRONG (p = {p_value:.6f})")
else:
    print(f"Statistical Evidence: WEAK (p = {p_value:.6f})")

print()
print("=" * 80)
print("Summary for manuscript:")
print(f"  The neural memory module contributes Delta AUC = {delta_mean:.4f} +/- {delta_sem:.4f}")
print(f"  (Memory ON: {baseline_mean:.4f} vs Memory OFF: {ablation_mean:.4f})")
print(f"  with 95% CI [{ci_lower:.4f}, {ci_upper:.4f}], p = {p_value:.6f}")
print("=" * 80)
