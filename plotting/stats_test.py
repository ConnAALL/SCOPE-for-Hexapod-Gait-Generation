import os
import glob
import numpy as np
from numpy import trapz
from scipy import stats
from scipy.stats import binom
from datetime import datetime

def select_recent_files(folder, top_k=500):
    """Return paths to the most recent top_k .npy files based on timestamp in filename."""
    def extract_timestamp(name):
        try:
            parts = name.split('_')
            return datetime.strptime(parts[-2] + '_' + parts[-1].split('.')[0], '%y%m%d_%H%M%S')
        except Exception:
            return datetime.min

    files = sorted(glob.glob(os.path.join(folder, "*.npy")),
                   key=lambda x: extract_timestamp(os.path.basename(x)),
                   reverse=True)
    return files[:top_k]


def load_final_best(files):
    """Reuse the final best DistanceDifference from each run."""
    finals = []
    for fn in files:
        data = np.load(fn, allow_pickle=True).item()
        finals.append(data["DistanceDifference_best"][-1])
    return np.array(finals)

def compute_auc_array(files):
    """Return 1D array of AUC under the DistanceDifference_avg curve for each run."""
    aucs = []
    for fn in files:
        d = np.load(fn, allow_pickle=True).item()
        gens = d["Generation"]
        avg  = d["DistanceDifference_avg"]
        aucs.append(trapz(avg, gens))
    return np.array(aucs)


def cliffs_delta(x, y):
    """
    Compute Cliff's delta:
      (#pairs where x_i > y_j minus #pairs where x_i < y_j)
    """
    n, m = len(x), len(y)
    # pairwise comparisons
    greater = 0
    less    = 0
    for xi in x:
        greater += np.sum(xi > y)
        less    += np.sum(xi < y)
    return (greater - less) / (n * m), greater / (n * m)

def pairwise_median_diff_ci(x, y, ci=0.95):
    """
    Compute a deterministic percentile-based confidence interval for:
        median(y) - median(x)
    by evaluating all pairwise differences between elements in y and x.

    Parameters:
        x, y: 1D arrays (e.g., GA and LREP data)
        ci: confidence level between 0 and 1 (default 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound) of the CI
    """
    diffs = np.array([ly - gx for ly in y for gx in x])
    alpha = 1.0 - ci
    lower = np.percentile(diffs, 100 * (alpha / 2))
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return lower, upper

def run_comparison(name, ga_data, lrep_data):
    """
    Peform statistical tests and print the results.
    Tests performed are:
        - Mann–Whitney U test
        - Cliff's delta effect size
        - Bootstrap CI on median difference
    """
    print(f"\n{name} (GA vs. LREP):")
    print("======================================================================")
    # Mann–Whitney U:
    # In this test, there are two different cases
    # 1) H₀: median(LREP) ≤ median(GA) --> Null hypothesis --> LREP is not better than GA
    # 2) H₁: median(LREP) > median(GA) --> Alternative hypothesis --> LREP is better than GA
    # This test compares the distributions of the two groups
    u_stat, p_val = stats.mannwhitneyu(lrep_data, ga_data, alternative="greater")
    print(f"Mann–Whitney U statistic = {u_stat:.2f}, p-value = {p_val:.5e}")
    if p_val < 0.05:
        print("→ p < 0.05: reject H₀.  LREP performance is statistically greater than GA.")
    else:
        print("→ p ≥ 0.05: fail to reject H₀.  No significant evidence that LREP > GA.")

    print("======================================================================")

    # Cliff's delta test
    # It measures the effect size of the difference between two groups
    # (number of LREP > GA pairs - number of LREP < GA pairs) / total pairs
    c_delta = cliffs_delta(lrep_data, ga_data)
    print(f"Cliff's Delta = {c_delta[0]:.3f}")
    print("Odds of LREP > GA pairs = {:.3f}".format(100 * c_delta[1]))

    print("======================================================================")

    # 1) Compute 95% CIs for each group’s mean
    alpha = 0.05

    # GA
    n_ga     = len(ga_data)
    mean_ga  = np.mean(ga_data)
    se_ga    = np.std(ga_data, ddof=1) / np.sqrt(n_ga)
    t_ga     = stats.t.ppf(1 - alpha/2, df=n_ga - 1)
    ci_ga_lo = mean_ga - t_ga * se_ga
    ci_ga_hi = mean_ga + t_ga * se_ga

    # LREP
    n_lr     = len(lrep_data)
    mean_lr  = np.mean(lrep_data)
    se_lr    = np.std(lrep_data, ddof=1) / np.sqrt(n_lr)
    t_lr     = stats.t.ppf(1 - alpha/2, df=n_lr - 1)
    ci_lr_lo = mean_lr - t_lr * se_lr
    ci_lr_hi = mean_lr + t_lr * se_lr

    print(f"GA   mean 95% CI: [{ci_ga_lo:.4f}, {ci_ga_hi:.4f}]")
    print(f"LREP mean 95% CI: [{ci_lr_lo:.4f}, {ci_lr_hi:.4f}]")

    # 2) Compute intersection (overlap) of those two intervals
    over_lo   = max(ci_ga_lo, ci_lr_lo)
    over_hi   = min(ci_ga_hi, ci_lr_hi)
    over_len  = max(0.0, over_hi - over_lo)

    # 3) Express overlap as % of each CI’s width
    width_ga  = ci_ga_hi - ci_ga_lo
    width_lr  = ci_lr_hi - ci_lr_lo
    prop_ga   = over_len / width_ga if width_ga>0 else 0
    prop_lr   = over_len / width_lr if width_lr>0 else 0

    print(f"\nOverlap interval: [{over_lo:.4f}, {over_hi:.4f}]")
    print(f" → Overlap length = {over_len:.4f}")
    print(f" → {prop_ga*100:.1f}% of GA’s CI")
    print(f" → {prop_lr*100:.1f}% of LREP’s CI")

def main():
    """
    Main loop to perform the statistical tests between the GA and LREP
    """

    ga_dir   = os.path.join("processed_data", "SSGA")
    lrep_dir = os.path.join("processed_data", "DCT")

    ga_files   = select_recent_files(ga_dir, top_k=500)
    lrep_files = select_recent_files(lrep_dir, top_k=500)

    ga_final   = load_final_best(ga_files)
    lrep_final = load_final_best(lrep_files)

    # Compare the final values
    # Perform Mann-Whitney U test, Cliff's delta, and bootstrap CI
    run_comparison("Final-generation Best Distance Difference", ga_final, lrep_final)

    # 1) Build the full vector of pairwise differences:
#    D_ij = LREP[i] - GA[j]
    diffs = np.subtract.outer(lrep_final, ga_final).ravel()

    # proportion of pairwise diffs ≤ 0
    p0 = np.mean(diffs <= 0)

    # The maximal two-sided CI for which lower bound > 0 is
    #   CI_max = 1 − 2 * p0
    ci_max = 1.0 - 2*p0

    print(f"Maximum CI level at which lower‐bound > 0: {ci_max*100:.1f}%")
if __name__ == "__main__":
    main()
