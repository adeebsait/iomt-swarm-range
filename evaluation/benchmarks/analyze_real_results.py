"""
Analyze real experiment results
"""
import json
import numpy as np
from scipy import stats
from collections import defaultdict

# Load your actual results
results_file = 'evaluation/results/real_experiments/real_results_20251025_005800.json'
with open(results_file, 'r') as f:
    results = json.load(f)

print("="*80)
print("REAL EXPERIMENT RESULTS ANALYSIS")
print(f"Results file: {results_file}")
print(f"Total trials: {len(results)}")
print("="*80)

# Group by algorithm and device scale
summary = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for r in results:
    algo = r['algorithm']
    num_dev = r.get('num_devices', 3)
    
    summary[num_dev][algo]['accuracy'].append(r['accuracy'])
    summary[num_dev][algo]['f1_score'].append(r['f1_score'])
    summary[num_dev][algo]['recall'].append(r['recall'])
    summary[num_dev][algo]['precision'].append(r['precision'])
    summary[num_dev][algo]['latency'].append(r['avg_detection_latency'])

# Print summary by device scale
for num_devices in sorted(summary.keys()):
    print(f"\n{'='*80}")
    print(f"DEVICE SCALE: {num_devices} devices")
    print('='*80)
    
    for algo in sorted(summary[num_devices].keys()):
        data = summary[num_devices][algo]
        n_trials = len(data['accuracy'])
        
        print(f"\n{algo}:")
        print(f"  Accuracy:  {np.mean(data['accuracy'])*100:.2f}% ± {np.std(data['accuracy'])*100:.2f}%")
        print(f"  Precision: {np.mean(data['precision'])*100:.2f}% ± {np.std(data['precision'])*100:.2f}%")
        print(f"  Recall:    {np.mean(data['recall'])*100:.2f}% ± {np.std(data['recall'])*100:.2f}%")
        print(f"  F1-Score:  {np.mean(data['f1_score'])*100:.2f}% ± {np.std(data['f1_score'])*100:.2f}%")
        print(f"  Latency:   {np.mean(data['latency']):.3f}s ± {np.std(data['latency']):.3f}s")
        print(f"  Trials:    {n_trials}")

# Overall comparison (all device scales combined)
print(f"\n{'='*80}")
print("OVERALL COMPARISON (All Device Scales)")
print('='*80)

overall = defaultdict(lambda: defaultdict(list))
for num_dev in summary:
    for algo in summary[num_dev]:
        for metric in ['accuracy', 'f1_score', 'recall', 'latency']:
            overall[algo][metric].extend(summary[num_dev][algo][metric])

print()
for algo in sorted(overall.keys()):
    data = overall[algo]
    print(f"{algo:10s} | Acc: {np.mean(data['accuracy'])*100:.2f}% | "
          f"F1: {np.mean(data['f1_score'])*100:.2f}% | "
          f"Lat: {np.mean(data['latency']):.3f}s | "
          f"N={len(data['accuracy'])}")

# Statistical significance
print(f"\n{'='*80}")
print("STATISTICAL SIGNIFICANCE TESTING")
print('='*80)

if 'Hybrid' in overall and 'Baseline' in overall:
    hybrid_acc = overall['Hybrid']['accuracy']
    baseline_acc = overall['Baseline']['accuracy']
    
    t_stat, p_value = stats.ttest_ind(hybrid_acc, baseline_acc)
    
    print(f"\nHybrid vs Baseline (t-test):")
    print(f"  Hybrid mean:    {np.mean(hybrid_acc)*100:.2f}%")
    print(f"  Baseline mean:  {np.mean(baseline_acc)*100:.2f}%")
    print(f"  Difference:     {(np.mean(hybrid_acc) - np.mean(baseline_acc))*100:.2f}%")
    print(f"  t-statistic:    {t_stat:.4f}")
    print(f"  p-value:        {p_value:.6f}")
    
    if p_value < 0.001:
        sig = "*** (HIGHLY SIGNIFICANT)"
    elif p_value < 0.01:
        sig = "** (VERY SIGNIFICANT)"
    elif p_value < 0.05:
        sig = "* (SIGNIFICANT)"
    else:
        sig = "ns (NOT SIGNIFICANT)"
    print(f"  Significance:   {sig}")
    
    # Effect size
    mean_diff = np.mean(hybrid_acc) - np.mean(baseline_acc)
    pooled_std = np.sqrt((np.std(hybrid_acc)**2 + np.std(baseline_acc)**2) / 2)
    cohens_d = mean_diff / pooled_std
    
    effect = 'LARGE' if abs(cohens_d) > 0.8 else 'MEDIUM' if abs(cohens_d) > 0.5 else 'SMALL'
    print(f"  Cohen's d:      {cohens_d:.4f} ({effect} effect size)")

# ANOVA across all algorithms
print(f"\nANOVA (all algorithms):")
algo_accuracies = [overall[algo]['accuracy'] for algo in sorted(overall.keys())]
f_stat, p_value = stats.f_oneway(*algo_accuracies)
print(f"  F-statistic:    {f_stat:.4f}")
print(f"  p-value:        {p_value:.6f}")
if p_value < 0.05:
    print(f"  Result:         SIGNIFICANT - algorithms differ significantly")
else:
    print(f"  Result:         NOT SIGNIFICANT")

print(f"\n{'='*80}")
print("PUBLICATION-READY RESULTS ✓")
print('='*80)

# Save summary
summary_data = {}
for algo in sorted(overall.keys()):
    data = overall[algo]
    summary_data[algo] = {
        'accuracy_mean': float(np.mean(data['accuracy'])),
        'accuracy_std': float(np.std(data['accuracy'])),
        'f1_mean': float(np.mean(data['f1_score'])),
        'f1_std': float(np.std(data['f1_score'])),
        'latency_mean': float(np.mean(data['latency'])),
        'latency_std': float(np.std(data['latency'])),
        'n_trials': len(data['accuracy'])
    }

with open('evaluation/results/summary_analysis.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

print("\nSummary saved to: evaluation/results/summary_analysis.json")
