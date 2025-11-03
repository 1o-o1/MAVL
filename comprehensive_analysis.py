#!/usr/bin/env python3
"""
COMPREHENSIVE CHEXPERT ANALYSIS SUITE

Generates all metrics, tables, and visualizations for journal submission:
1. Prevalence analysis (disease distribution)
2. Per-pathology performance breakdown
3. Subgroup analysis (sex, age, view type)
4. Calibration analysis framework
5. LaTeX macros generation

Does NOT require:
- Ablation results (documented separately)
- Per-case predictions (uses epoch-level metrics)
- Grad-CAM execution (provides implementation plan)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
ROOT_DIR = Path("E:/Research/zero shot lung")
DATA_ROOT = ROOT_DIR / "CheXpert-v1.0-small"
RESULTS_DIR = ROOT_DIR / "results_full_scale_corrected"
OUTPUT_DIR = RESULTS_DIR  # Save outputs alongside existing results

# CheXpert pathology columns (official order)
PATHOLOGIES = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]

print("="*80)
print("CHEXPERT COMPREHENSIVE ANALYSIS SUITE")
print("="*80)
print()

# ============================================================================
# PHASE 1: LOAD BASELINE RESULTS
# ============================================================================

print("[1/7] Loading baseline results...")
with open(RESULTS_DIR / 'summary_results.json', 'r') as f:
    baseline = json.load(f)

seeds = [42, 123, 456]
print(f"  Loaded results for {len(baseline['all_results'])} seeds")
print(f"  Mean AUROC: {baseline['auroc_mean']:.4f} ± {baseline['auroc_std']:.4f}")
print()

# ============================================================================
# PHASE 2: PREVALENCE ANALYSIS
# ============================================================================

print("[2/7] Computing disease prevalence...")

# Load validation CSV (this is our "test" set)
valid_df = pd.read_csv(DATA_ROOT / 'valid.csv')
print(f"  Validation set size: {len(valid_df)} images")

# Handle uncertain labels (-1 -> 0, NaN -> 0)
prevalence_data = []

for pathology in PATHOLOGIES:
    if pathology in valid_df.columns:
        # Replace -1 and NaN with 0, then count positives
        labels = valid_df[pathology].fillna(0).replace(-1, 0)
        positive_count = (labels == 1.0).sum()
        total_count = len(labels)
        prevalence_pct = (positive_count / total_count) * 100 if total_count > 0 else 0

        prevalence_data.append({
            'pathology': pathology,
            'positive_count': int(positive_count),
            'total_count': int(total_count),
            'prevalence_pct': round(prevalence_pct, 2)
        })
    else:
        prevalence_data.append({
            'pathology': pathology,
            'positive_count': 0,
            'total_count': len(valid_df),
            'prevalence_pct': 0.0
        })

prevalence_df = pd.DataFrame(prevalence_data)
prevalence_df.to_csv(OUTPUT_DIR / 'prevalence.csv', index=False)
print(f"  Saved: prevalence.csv")
print(f"  Most common: {prevalence_df.iloc[prevalence_df['positive_count'].argmax()]['pathology']} ({prevalence_df['positive_count'].max()} cases)")
print()

# ============================================================================
# PHASE 3: PER-PATHOLOGY PERFORMANCE
# ============================================================================

print("[3/7] Extracting per-pathology performance...")

# From baseline results, compute per-pathology metrics
# Note: Baseline results have overall AUROC, not per-pathology
# We'll document what WOULD be computed with per-case predictions

per_pathology_template = []
for i, pathology in enumerate(PATHOLOGIES):
    per_pathology_template.append({
        'pathology': pathology,
        'prevalence_pct': prevalence_df.iloc[i]['prevalence_pct'],
        'auroc_mean': np.nan,  # Would compute from per-case predictions
        'auroc_std': np.nan,
        'auprc_mean': np.nan,
        'f1_mean': np.nan,
        'note': 'Requires per-case predictions'
    })

per_pathology_df = pd.DataFrame(per_pathology_template)
per_pathology_df.to_csv(OUTPUT_DIR / 'per_pathology_performance.csv', index=False)
print(f"  Saved: per_pathology_performance.csv (template)")
print(f"  Note: Per-pathology metrics require per-case predictions (not available)")
print()

# ============================================================================
# PHASE 4: SUBGROUP ANALYSIS
# ============================================================================

print("[4/7] Analyzing demographic subgroups...")

# Analyze by available metadata
subgroup_results = []

# Sex distribution
if 'Sex' in valid_df.columns:
    sex_counts = valid_df['Sex'].value_counts()
    for sex, count in sex_counts.items():
        subgroup_results.append({
            'subgroup_type': 'sex',
            'subgroup_value': sex,
            'n_samples': int(count),
            'pct_of_total': round(count / len(valid_df) * 100, 2),
            'auroc_mean': np.nan,
            'auroc_ci_lo': np.nan,
            'auroc_ci_hi': np.nan,
            'note': 'Requires stratified per-case predictions'
        })

# Age distribution (binned)
if 'Age' in valid_df.columns:
    valid_df['age_bin'] = pd.cut(valid_df['Age'], bins=[0, 40, 60, 80, 120], labels=['<40', '40-60', '60-80', '80+'])
    age_counts = valid_df['age_bin'].value_counts()
    for age_bin, count in age_counts.items():
        subgroup_results.append({
            'subgroup_type': 'age',
            'subgroup_value': str(age_bin),
            'n_samples': int(count),
            'pct_of_total': round(count / len(valid_df) * 100, 2),
            'auroc_mean': np.nan,
            'auroc_ci_lo': np.nan,
            'auroc_ci_hi': np.nan,
            'note': 'Requires stratified per-case predictions'
        })

# View type distribution
if 'Frontal/Lateral' in valid_df.columns:
    view_counts = valid_df['Frontal/Lateral'].value_counts()
    for view, count in view_counts.items():
        subgroup_results.append({
            'subgroup_type': 'view',
            'subgroup_value': view,
            'n_samples': int(count),
            'pct_of_total': round(count / len(valid_df) * 100, 2),
            'auroc_mean': np.nan,
            'auroc_ci_lo': np.nan,
            'auroc_ci_hi': np.nan,
            'note': 'Requires stratified per-case predictions'
        })

subgroup_df = pd.DataFrame(subgroup_results)
subgroup_df.to_csv(OUTPUT_DIR / 'subgroup_metrics.csv', index=False)
print(f"  Saved: subgroup_metrics.csv")
print(f"  Sex distribution: {sex_counts.to_dict()}")
print(f"  View types: {view_counts.to_dict()}")
print()

# ============================================================================
# PHASE 5: CALIBRATION ANALYSIS FRAMEWORK
# ============================================================================

print("[5/7] Creating calibration analysis framework...")

calibration_doc = {
    "method": "Temperature scaling (Platt scaling)",
    "description": "Fit single scalar T on validation set to calibrate model outputs",
    "status": "Framework defined, execution requires per-case predictions",
    "metrics_to_compute": {
        "ECE": "Expected Calibration Error (mean |confidence - accuracy| across bins)",
        "MCE": "Maximum Calibration Error (max bin |confidence - accuracy|)",
        "NLL": "Negative Log-Likelihood",
        "temperature_T": "Optimal temperature scalar"
    },
    "per_class_thresholds": {
        "Youden_J": "max(TPR - FPR) for optimal sensitivity-specificity tradeoff",
        "F1_optimal": "threshold maximizing F1 score"
    },
    "implementation_notes": [
        "Use scipy.optimize.minimize to fit T on validation NLL",
        "Compute reliability diagram: 10 bins, plot mean confidence vs accuracy",
        "Apply per-class thresholds to test set predictions",
        "Report macro-averaged F1 before/after thresholding"
    ],
    "required_inputs": {
        "validation_predictions": "Per-case predicted probabilities [N, 14]",
        "validation_labels": "Binary labels [N, 14]",
        "test_predictions": "Per-case predicted probabilities [N_test, 14]",
        "test_labels": "Binary labels [N_test, 14]"
    },
    "expected_outputs": [
        "calibration_metrics.json (ECE, MCE, NLL, T)",
        "thresholds_per_class.csv (class, youden_threshold, f1opt_threshold, test_f1)",
        "reliability_diagram_before.png",
        "reliability_diagram_after.png"
    ]
}

with open(OUTPUT_DIR / 'calibration_framework.json', 'w') as f:
    json.dump(calibration_doc, f, indent=2)
print(f"  Saved: calibration_framework.json")
print(f"  Note: Execution deferred pending per-case prediction extraction")
print()

# ============================================================================
# PHASE 6: GRAD-CAM VISUALIZATION PLAN
# ============================================================================

print("[6/7] Creating Grad-CAM visualization plan...")

gradcam_plan = {
    "objective": "Qualitative explainability via Grad-CAM heatmaps",
    "dataset": "CheXpert (no bbox ground truth available)",
    "limitation": "CheXpert lacks localization annotations; quantitative IoU metrics reserved for VinDr-CXR",
    "qualitative_approach": {
        "method": "Grad-CAM on final conv layer of ViT-B/16 encoder",
        "sample_selection": [
            "5 high-confidence correct predictions (e.g., Pneumonia)",
            "5 high-confidence incorrect predictions (misclassifications)",
            "5 normal cases (No Finding = 1)"
        ],
        "visualization": "3×5 grid PNG with original image + heatmap overlay"
    },
    "implementation_steps": [
        "1. Load trained model checkpoint (best seed, e.g., seed 456)",
        "2. Select 15 diverse validation images based on predictions",
        "3. Use pytorch-grad-cam library: GradCAM(model=vision_encoder, target_layer=encoder.layer[-1])",
        "4. Generate heatmaps for target pathology class",
        "5. Overlay heatmaps on original images with alpha=0.5",
        "6. Create matplotlib 3×5 grid figure",
        "7. Save as chexpert_gradcam_qualitative_grid.png"
    ],
    "code_snippet": """
# Pseudo-code for Grad-CAM generation
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

model.eval()
target_layer = model.vision_encoder.encoder.layers[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

for img_path in selected_images:
    img_tensor = load_and_preprocess(img_path)
    grayscale_cam = cam(input_tensor=img_tensor, targets=[target_class])
    visualization = show_cam_on_image(rgb_img, grayscale_cam)
    save_image(visualization, output_path)
""",
    "status": "Plan documented, execution deferred",
    "estimated_time": "~2 hours (requires model checkpoint loading + inference)"
}

with open(OUTPUT_DIR / 'gradcam_plan.json', 'w') as f:
    json.dump(gradcam_plan, f, indent=2)
print(f"  Saved: gradcam_plan.json")
print()

# ============================================================================
# PHASE 7: LATEX MACROS GENERATION
# ============================================================================

print("[7/7] Generating LaTeX macros...")

macros = {}

# Baseline metrics (mean across seeds)
macros['baselineAUROCmean'] = f"{baseline['auroc_mean']:.3f}"
macros['baselineAUROCstd'] = f"{baseline['auroc_std']:.4f}"

# Per-seed metrics
for i, seed in enumerate(seeds):
    result = baseline['all_results'][i]
    seed_label = seed
    macros[f'seed{seed_label}BestAUROC'] = f"{result['best_auroc']:.4f}"
    macros[f'seed{seed_label}FinalAUROC'] = f"{result['final_auroc']:.4f}"
    macros[f'seed{seed_label}FinalF1'] = f"{result['final_f1']:.4f}"
    macros[f'seed{seed_label}FinalAUPRC'] = f"{result['final_auprc']:.4f}"
    macros[f'seed{seed_label}TrainingHours'] = f"{result['elapsed_time'] / 3600:.2f}"

# Dataset statistics
macros['validationSetSize'] = f"{len(valid_df)}"
macros['numPathologies'] = f"{len(PATHOLOGIES)}"
macros['numEpochs'] = "40"
macros['batchSize'] = "8"
macros['learningRate'] = "1e-4"
macros['memorySlots'] = "256"
macros['memoryDim'] = "768"

# Prevalence statistics
macros['mostCommonPathology'] = prevalence_df.iloc[prevalence_df['positive_count'].argmax()]['pathology'].replace(' ', '')
macros['mostCommonPrevalence'] = f"{prevalence_df['prevalence_pct'].max():.2f}"
macros['leastCommonPathology'] = prevalence_df.iloc[prevalence_df['positive_count'].argmin()]['pathology'].replace(' ', '')
macros['leastCommonPrevalence'] = f"{prevalence_df['prevalence_pct'].min():.2f}"

# Demographic statistics
if not subgroup_df.empty:
    sex_male_pct = subgroup_df[(subgroup_df['subgroup_type'] == 'sex') & (subgroup_df['subgroup_value'] == 'Male')]['pct_of_total'].values
    if len(sex_male_pct) > 0:
        macros['sexMalePercent'] = f"{sex_male_pct[0]:.1f}"
    sex_female_pct = subgroup_df[(subgroup_df['subgroup_type'] == 'sex') & (subgroup_df['subgroup_value'] == 'Female')]['pct_of_total'].values
    if len(sex_female_pct) > 0:
        macros['sexFemalePercent'] = f"{sex_female_pct[0]:.1f}"

# Ablation placeholders (to be filled when ablation runs)
macros['ablationAUROCmean'] = "TBD"
macros['memoryDeltaAUC'] = "TBD"
macros['memoryDeltaPvalue'] = "TBD"

# Generate LaTeX commands
latex_macros_text = "% CheXpert Experiment LaTeX Macros\n"
latex_macros_text += "% Auto-generated: " + datetime.now().isoformat() + "\n\n"

for key, value in sorted(macros.items()):
    latex_macros_text += f"\\newcommand{{\\{key}}}{{{value}}}\n"

with open(OUTPUT_DIR / 'latex_macros.tex', 'w') as f:
    f.write(latex_macros_text)

print(f"  Saved: latex_macros.tex ({len(macros)} macros)")
print()

# ============================================================================
# PHASE 8: COMPREHENSIVE JSON SUMMARY
# ============================================================================

print("[8/7] Creating comprehensive JSON summary...")

comprehensive_results = {
    "metadata": {
        "generated_at": datetime.now().isoformat(),
        "dataset": "CheXpert-v1.0-small",
        "validation_set_size": len(valid_df),
        "num_pathologies": len(PATHOLOGIES),
        "seeds": seeds
    },
    "baseline_memory_on": {
        "mean_best_auroc": baseline['auroc_mean'],
        "std_best_auroc": baseline['auroc_std'],
        "per_seed_results": [
            {
                "seed": result['seed'],
                "best_auroc": result['best_auroc'],
                "final_auroc": result['final_auroc'],
                "final_f1": result['final_f1'],
                "final_auprc": result['final_auprc'],
                "training_time_hours": result['elapsed_time'] / 3600
            }
            for result in baseline['all_results']
        ]
    },
    "ablation_memory_off": {
        "status": "script_prepared_not_executed",
        "estimated_runtime_hours": 33,
        "script_path": "full_scale_training_ablation.py",
        "documentation": "ABLATION_DOCUMENTATION.md"
    },
    "prevalence": {
        "data": prevalence_data,
        "statistics": {
            "most_common": {
                "pathology": prevalence_df.iloc[prevalence_df['positive_count'].argmax()]['pathology'],
                "count": int(prevalence_df['positive_count'].max()),
                "prevalence_pct": float(prevalence_df['prevalence_pct'].max())
            },
            "least_common": {
                "pathology": prevalence_df.iloc[prevalence_df['positive_count'].argmin()]['pathology'],
                "count": int(prevalence_df['positive_count'].min()),
                "prevalence_pct": float(prevalence_df['prevalence_pct'].min())
            }
        }
    },
    "subgroups": {
        "status": "demographics_extracted_metrics_pending",
        "available_metadata": ["Sex", "Age", "Frontal/Lateral", "AP/PA"],
        "subgroup_counts": subgroup_results,
        "note": "Per-subgroup AUROC requires stratified per-case predictions"
    },
    "calibration": {
        "status": "framework_defined",
        "framework_file": "calibration_framework.json",
        "note": "Execution requires per-case predictions"
    },
    "explainability": {
        "qualitative": {
            "method": "Grad-CAM",
            "status": "plan_documented",
            "plan_file": "gradcam_plan.json"
        },
        "quantitative": {
            "method": "IoU with ground-truth bboxes",
            "status": "not_applicable_chexpert",
            "note": "CheXpert lacks bbox annotations; reserved for VinDr-CXR"
        }
    },
    "artifacts": {
        "prevalence_csv": "prevalence.csv",
        "subgroup_metrics_csv": "subgroup_metrics.csv",
        "per_pathology_csv": "per_pathology_performance.csv",
        "latex_macros": "latex_macros.tex",
        "calibration_framework": "calibration_framework.json",
        "gradcam_plan": "gradcam_plan.json",
        "ablation_script": "../full_scale_training_ablation.py",
        "ablation_docs": "ABLATION_DOCUMENTATION.md"
    }
}

with open(OUTPUT_DIR / 'chexpert_comprehensive_results.json', 'w') as f:
    json.dump(comprehensive_results, f, indent=2, default=str)

print(f"  Saved: chexpert_comprehensive_results.json")
print()

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print()
print("Generated files:")
print(f"  1. prevalence.csv - Disease prevalence in validation set")
print(f"  2. per_pathology_performance.csv - Template for per-pathology metrics")
print(f"  3. subgroup_metrics.csv - Demographic subgroup distribution")
print(f"  4. calibration_framework.json - Calibration analysis plan")
print(f"  5. gradcam_plan.json - Explainability visualization plan")
print(f"  6. latex_macros.tex - {len(macros)} LaTeX macros")
print(f"  7. chexpert_comprehensive_results.json - Master summary")
print()
print("Next steps:")
print("  - Execute ablation: python full_scale_training_ablation.py (33 hours)")
print("  - Extract per-case predictions for calibration + subgroup analysis")
print("  - Generate Grad-CAM visualizations")
print("  - Run DeLong paired test (memory ON vs OFF)")
print()
