"""
DeLong Test for Paired ROC Curves - Memory Ablation Analysis
============================================================

Statistical test to quantify the contribution of the neural memory module.
Tests whether AUROC difference between memory-ON and memory-OFF is significant.

Reference:
  DeLong et al. (1988). Comparing the areas under two or more correlated
  receiver operating characteristic curves: a nonparametric approach.
  Biometrics, 44(3), 837-845.

Author: Research Team
Date: 2025-11-03
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging
from scipy import stats
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AUROCMetrics:
    """Container for AUROC metrics per seed"""
    seed: int
    auroc: float
    auroc_std: float
    f1: float
    auprc: float
    training_hours: float


class DeLongTest:
    """
    Implements the DeLong test for paired ROC curves.

    This test determines if two correlated ROC curves (from the same test set)
    have significantly different areas under the curve (AUC).

    For memory ablation:
    - ROC curve 1: Memory ON (baseline)
    - ROC curve 2: Memory OFF (ablation)
    - Per-case predictions: Binary classification scores for each image
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize DeLong test.

        Args:
            alpha: Significance level (default 0.05 for 95% CI)
        """
        self.alpha = alpha
        self.z_critical = stats.norm.ppf(1 - alpha/2)

    def compute_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        Compute AUC from binary labels and continuous scores.

        Args:
            y_true: Binary labels (0 or 1)
            y_scores: Continuous prediction scores

        Returns:
            AUC value
        """
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_scores)

    def compute_auc_ci(self, y_true: np.ndarray, y_scores: np.ndarray,
                       n_bootstrap: int = 5000) -> Tuple[float, float, float]:
        """
        Compute AUC with 95% confidence interval via bootstrap.

        Args:
            y_true: Binary labels
            y_scores: Prediction scores
            n_bootstrap: Number of bootstrap resamples

        Returns:
            (auc, ci_lower, ci_upper)
        """
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(y_true, y_scores)

        # Bootstrap resampling
        n = len(y_true)
        aucs = []
        np.random.seed(42)
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            auc_boot = roc_auc_score(y_true[indices], y_scores[indices])
            aucs.append(auc_boot)

        aucs = np.array(aucs)
        ci_lower = np.percentile(aucs, 2.5)
        ci_upper = np.percentile(aucs, 97.5)

        return auc, ci_lower, ci_upper

    def delong_test_paired(self, y_true: np.ndarray,
                           y_scores_on: np.ndarray,
                           y_scores_off: np.ndarray) -> Dict:
        """
        Perform paired DeLong test comparing two ROC curves.

        Args:
            y_true: True binary labels (same for both models)
            y_scores_on: Prediction scores from memory-ON model
            y_scores_off: Prediction scores from memory-OFF model

        Returns:
            Dictionary with test results
        """
        from sklearn.metrics import roc_auc_score

        # Compute AUCs
        auc_on, ci_on_low, ci_on_high = self.compute_auc_ci(y_true, y_scores_on)
        auc_off, ci_off_low, ci_off_high = self.compute_auc_ci(y_true, y_scores_off)

        delta_auc = auc_on - auc_off

        # Simplified paired t-test on per-case AUCs
        # (Full DeLong requires per-case AUC computation)
        n = len(y_true)

        # Compute per-sample contributions to AUC (approximation)
        # For proper DeLong, would need to compute θ values per threshold
        differences = np.abs(y_scores_on - y_scores_off)

        # Paired comparison using bootstrap on difference
        delta_aucs = []
        np.random.seed(42)
        for _ in range(5000):
            indices = np.random.choice(n, size=n, replace=True)
            auc_on_boot = roc_auc_score(y_true[indices], y_scores_on[indices])
            auc_off_boot = roc_auc_score(y_true[indices], y_scores_off[indices])
            delta_aucs.append(auc_on_boot - auc_off_boot)

        delta_aucs = np.array(delta_aucs)
        delta_ci_low = np.percentile(delta_aucs, 2.5)
        delta_ci_high = np.percentile(delta_aucs, 97.5)

        # Approximate p-value from bootstrap
        p_value = np.mean(np.abs(delta_aucs) >= np.abs(delta_auc))
        if p_value == 0:
            p_value = 1 / 5001  # Smallest possible p-value with 5000 resamples

        return {
            'auc_memory_on': auc_on,
            'ci_on_lower': ci_on_low,
            'ci_on_upper': ci_on_high,
            'auc_memory_off': auc_off,
            'ci_off_lower': ci_off_low,
            'ci_off_upper': ci_off_high,
            'delta_auc': delta_auc,
            'delta_ci_lower': delta_ci_low,
            'delta_ci_upper': delta_ci_high,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'n_samples': n
        }


class AblationResultsProcessor:
    """Process and analyze ablation study results"""

    def __init__(self,
                 baseline_dir: Path = Path("/e/Research/zero shot lung/results_full_scale_corrected"),
                 ablation_dir: Path = Path("/e/Research/zero shot lung/results_full_scale_ablation_fast")):
        """
        Initialize processor.

        Args:
            baseline_dir: Directory containing memory-ON results
            ablation_dir: Directory containing memory-OFF results
        """
        self.baseline_dir = baseline_dir
        self.ablation_dir = ablation_dir
        self.delong = DeLongTest()

    def load_per_seed_results(self, seed: int, is_ablation: bool = False) -> Dict:
        """
        Load per-seed results from JSON file.

        Args:
            seed: Random seed (42, 123, or 456)
            is_ablation: If True, load from ablation dir; else baseline dir

        Returns:
            Dictionary with seed results
        """
        results_dir = self.ablation_dir if is_ablation else self.baseline_dir
        results_file = results_dir / f"seed_{seed}_results.json"

        if not results_file.exists():
            logger.warning(f"Results file not found: {results_file}")
            return None

        with open(results_file) as f:
            return json.load(f)

    def load_summary_results(self, is_ablation: bool = False) -> Dict:
        """
        Load summary results across all seeds.

        Args:
            is_ablation: If True, load from ablation dir; else baseline dir

        Returns:
            Summary results dictionary
        """
        results_dir = self.ablation_dir if is_ablation else self.baseline_dir
        results_file = results_dir / "summary_results.json"

        if not results_file.exists():
            logger.warning(f"Summary file not found: {results_file}")
            return None

        with open(results_file) as f:
            return json.load(f)

    def extract_seed_metrics(self) -> Dict[str, Dict]:
        """
        Extract final metrics for each seed from both baseline and ablation.

        Returns:
            Dictionary mapping seed to metrics comparison
        """
        seeds = [42, 123, 456]
        results = {}

        for seed in seeds:
            logger.info(f"Extracting metrics for seed {seed}...")

            baseline = self.load_per_seed_results(seed, is_ablation=False)
            ablation = self.load_per_seed_results(seed, is_ablation=True)

            if baseline is None or ablation is None:
                logger.warning(f"Incomplete results for seed {seed}")
                continue

            results[seed] = {
                'baseline': {
                    'auroc': baseline.get('final_auroc', baseline.get('auroc')),
                    'f1': baseline.get('final_f1', baseline.get('f1')),
                    'auprc': baseline.get('final_auprc', baseline.get('auprc'))
                },
                'ablation': {
                    'auroc': ablation.get('final_auroc', ablation.get('auroc')),
                    'f1': ablation.get('final_f1', ablation.get('f1')),
                    'auprc': ablation.get('final_auprc', ablation.get('auprc'))
                }
            }

            # Compute delta
            results[seed]['delta'] = {
                'auroc': results[seed]['baseline']['auroc'] - results[seed]['ablation']['auroc'],
                'f1': results[seed]['baseline']['f1'] - results[seed]['ablation']['f1'],
                'auprc': results[seed]['baseline']['auprc'] - results[seed]['ablation']['auprc']
            }

        return results

    def compute_aggregate_metrics(self, seed_results: Dict) -> Dict:
        """
        Compute aggregate metrics across seeds.

        Args:
            seed_results: Per-seed results dictionary

        Returns:
            Aggregate metrics
        """
        seeds = list(seed_results.keys())

        baseline_aurocs = [seed_results[s]['baseline']['auroc'] for s in seeds]
        ablation_aurocs = [seed_results[s]['ablation']['auroc'] for s in seeds]
        delta_aurocs = [seed_results[s]['delta']['auroc'] for s in seeds]

        return {
            'baseline': {
                'auroc_mean': np.mean(baseline_aurocs),
                'auroc_std': np.std(baseline_aurocs),
                'auroc_sem': np.std(baseline_aurocs) / np.sqrt(len(seeds))
            },
            'ablation': {
                'auroc_mean': np.mean(ablation_aurocs),
                'auroc_std': np.std(ablation_aurocs),
                'auroc_sem': np.std(ablation_aurocs) / np.sqrt(len(seeds))
            },
            'delta': {
                'auroc_mean': np.mean(delta_aurocs),
                'auroc_std': np.std(delta_aurocs),
                'auroc_sem': np.std(delta_aurocs) / np.sqrt(len(seeds)),
                'auroc_values': delta_aurocs
            },
            'n_seeds': len(seeds)
        }

    def interpret_results(self, delta_auc: float, p_value: float) -> str:
        """
        Interpret ablation results in scientific terms.

        Args:
            delta_auc: Difference in AUC (memory ON - memory OFF)
            p_value: p-value from statistical test

        Returns:
            Interpretation string
        """
        if p_value < 0.05:
            if delta_auc >= 0.02:
                return "CRITICAL: Memory module contributes significantly (ΔAUC ≥ 0.02, p < 0.05)"
            elif delta_auc >= 0.01:
                return "IMPORTANT: Memory module has meaningful contribution (0.01 ≤ ΔAUC < 0.02, p < 0.05)"
            else:
                return "MARGINAL: Memory module has slight but significant effect (ΔAUC < 0.01, p < 0.05)"
        else:
            if delta_auc < 0.005:
                return "NEGLIGIBLE: Memory module effect not statistically significant (ΔAUC < 0.005, p > 0.05)"
            else:
                return "INSUFFICIENT EVIDENCE: Effect magnitude notable but not significant (p > 0.05)"

    def generate_report(self, output_file: Path) -> None:
        """
        Generate comprehensive ablation analysis report.

        Args:
            output_file: Path to save report
        """
        logger.info("Generating ablation analysis report...")

        # Extract results
        seed_results = self.extract_seed_metrics()
        agg_metrics = self.compute_aggregate_metrics(seed_results)

        # Generate report
        report = []
        report.append("=" * 80)
        report.append("MEMORY ABLATION STUDY - STATISTICAL ANALYSIS")
        report.append("CheXpert Validation Set - 3 Seeds × 40 Epochs")
        report.append("=" * 80)
        report.append("")

        # Per-seed results
        report.append("PER-SEED RESULTS")
        report.append("-" * 80)
        report.append(f"{'Seed':<10} {'Memory ON':<20} {'Memory OFF':<20} {'ΔAUC':<20}")
        report.append("-" * 80)

        for seed, results in sorted(seed_results.items()):
            on_auc = results['baseline']['auroc']
            off_auc = results['ablation']['auroc']
            delta = results['delta']['auroc']
            report.append(f"{seed:<10} {on_auc:<20.4f} {off_auc:<20.4f} {delta:+.4f}")

        report.append("")

        # Aggregate results
        report.append("AGGREGATE RESULTS (3 SEEDS)")
        report.append("-" * 80)

        baseline_mean = agg_metrics['baseline']['auroc_mean']
        baseline_std = agg_metrics['baseline']['auroc_std']
        ablation_mean = agg_metrics['ablation']['auroc_mean']
        ablation_std = agg_metrics['ablation']['auroc_std']
        delta_mean = agg_metrics['delta']['auroc_mean']
        delta_std = agg_metrics['delta']['auroc_std']

        report.append(f"Memory ON (baseline):     {baseline_mean:.4f} ± {baseline_std:.4f}")
        report.append(f"Memory OFF (ablation):    {ablation_mean:.4f} ± {ablation_std:.4f}")
        report.append(f"ΔAUC (ON - OFF):          {delta_mean:+.4f} ± {delta_std:.4f}")
        report.append("")

        # Statistical interpretation
        if delta_mean >= 0.02:
            interpretation = "CRITICAL: Memory is essential for performance"
        elif delta_mean >= 0.01:
            interpretation = "IMPORTANT: Memory provides meaningful improvement"
        elif delta_mean >= 0.005:
            interpretation = "MARGINAL: Memory has modest positive effect"
        else:
            interpretation = "NEGLIGIBLE: Memory contribution is minimal"

        report.append(f"Interpretation: {interpretation}")
        report.append("")

        # Next steps for full DeLong test
        report.append("NEXT STEPS (AFTER PER-CASE PREDICTIONS AVAILABLE)")
        report.append("-" * 80)
        report.append("1. Extract per-image prediction scores from both models")
        report.append("2. Run paired DeLong test on ROC curves")
        report.append("3. Compute 95% CI on ΔAUC via bootstrap (5000 resamples)")
        report.append("4. Report p-value and statistical significance")
        report.append("")

        # Save report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))

        logger.info(f"Report saved to {output_file}")
        print('\n'.join(report))


def main():
    """Main execution"""
    processor = AblationResultsProcessor()

    # Check if ablation has completed
    summary_ablation = processor.load_summary_results(is_ablation=True)
    summary_baseline = processor.load_summary_results(is_ablation=False)

    if summary_ablation is None:
        logger.error("Ablation study not yet complete. Please wait for training to finish.")
        return

    # Generate report
    output_file = Path("/e/Research/zero shot lung/results_full_scale_corrected/delong_analysis_report.txt")
    processor.generate_report(output_file)


if __name__ == "__main__":
    main()
