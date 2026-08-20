"""Step 7: turn the raw results into numbers you can defend.

    python scripts/07_analyze.py

Writes outputs/analysis/summary.md plus two CSV files.

TWO IDEAS DO ALL THE WORK HERE
------------------------------
1. A success rate on 300 episodes is not exact. "50%" really means "somewhere
   around 50%", and we need to say how wide "around" is. That is the confidence
   interval. Without it, a 4-point difference looks like a result when it is
   usually noise.

2. Because every condition saw the SAME starting scenes in the same order, we
   can do much better than comparing two averages. We can look at each scene and
   ask "did A succeed where B failed?". Only the scenes where they disagree
   carry any information, and counting those is much more sensitive. That is
   McNemar's test.

The p-value answers one specific question: if the two conditions were really
equally good, how often would we see a split this lopsided by luck? Small means
rarely, so the difference is probably real.
"""

import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config                                      # noqa: E402


def wilson_interval(successes, total):
    """A 95% confidence interval for a success rate, as percentages.

    We use the Wilson formula rather than the textbook one because the simple
    version behaves badly near 0% and 100%, and can even suggest impossible
    rates like -3%.
    """
    if total == 0:
        return 0.0, 0.0
    z = 1.96                                   # 95%
    rate = successes / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return 100 * (centre - spread), 100 * (centre + spread)


def mcnemar_p_value(a_only, b_only):
    """Exact two-sided McNemar test.

    a_only = scenes where A succeeded and B failed
    b_only = scenes where B succeeded and A failed

    Scenes where both did the same thing tell us nothing about which is better,
    so they are not counted. If the two are equally good, each disagreement
    should be a coin flip, and this works out how surprising the observed split
    would be.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0

    smaller = min(a_only, b_only)
    # Probability of a split at least this lopsided, counting both directions.
    total = 0.0
    for k in range(smaller + 1):
        total += math.comb(n, k)
    p = 2.0 * total / (2 ** n)
    return min(1.0, p)


def load_results():
    """Read every finished evaluation. -> {condition: {seed: [True/False, ...]}}"""
    results = {}
    pattern = os.path.join(config.EVAL_DIR, "*_seed*", "eval_info.json")
    for path in sorted(glob.glob(pattern)):
        folder = os.path.basename(os.path.dirname(path))
        condition, _, seed_text = folder.rpartition("_seed")
        with open(path) as f:
            data = json.load(f)

        # Per-episode outcomes, keyed by (task, episode) so that two conditions
        # can be lined up scene by scene. The ORDER matters here: pairing only
        # works if episode 3 of task 5 in one condition is the same starting
        # scene as episode 3 of task 5 in another, which is exactly what
        # --env.init_states=true guarantees.
        outcomes = {}
        for task in data.get("per_task", []):
            task_id = task["task_id"]
            for i, success in enumerate(task["metrics"]["successes"]):
                outcomes[(task_id, i)] = bool(success)

        if not outcomes:
            print("WARNING: no per-episode results in %s" % path)
            continue

        results.setdefault(condition, {})[int(seed_text)] = outcomes
    return results


def main():
    results = load_results()
    if not results:
        print("no results yet. Run scripts/05_sweep.py first.")
        return 1

    os.makedirs(config.ANALYSIS_DIR, exist_ok=True)

    # --- success rate per condition ---
    pooled = {}
    for condition in results:
        outcomes = {}
        for seed in sorted(results[condition]):
            for key, success in results[condition][seed].items():
                outcomes[(seed,) + key] = success
        pooled[condition] = outcomes

    lines = ["# Ablation results", "", "## Success rate", "",
             "| condition | episodes | success | 95% confidence |",
             "|---|---|---|---|"]
    print("%-14s %9s %9s   95%% confidence" % ("condition", "episodes", "success"))
    def rate_of(condition):
        values = list(pooled[condition].values())
        return sum(values) / len(values) if values else 0.0

    for condition in sorted(pooled, key=lambda c: -rate_of(c)):
        outcomes = list(pooled[condition].values())
        successes = sum(outcomes)
        total = len(outcomes)
        rate = 100.0 * successes / total
        low, high = wilson_interval(successes, total)
        print("%-14s %9d %8.1f%%   [%.1f, %.1f]" % (condition, total, rate, low, high))
        lines.append("| %s | %d | %.1f%% | [%.1f, %.1f] |"
                     % (condition, total, rate, low, high))

    # --- paired comparisons against the no-conditioning floor ---
    baseline = "C_none"
    comparisons = []
    if baseline in pooled:
        lines += ["", "## Compared with %s, scene by scene" % baseline, "",
                  "| condition | better here | worse here | p-value | different? |",
                  "|---|---|---|---|---|"]
        print("\n%-14s %11s %10s %9s  %s"
              % ("condition", "better", "worse", "p-value", "different?"))
        for condition in sorted(pooled):
            if condition == baseline:
                continue
            a = pooled[condition]
            b = pooled[baseline]
            # Compare only the scenes BOTH conditions actually ran.
            shared = sorted(set(a) & set(b))
            n = len(shared)
            if n == 0:
                continue
            a_only = sum(1 for key in shared if a[key] and not b[key])
            b_only = sum(1 for key in shared if b[key] and not a[key])
            p = mcnemar_p_value(a_only, b_only)
            verdict = "yes" if p < 0.05 else "no"
            print("%-14s %11d %10d %9.4f  %s" % (condition, a_only, b_only, p, verdict))
            lines.append("| %s | %d | %d | %.4f | %s |" % (condition, a_only, b_only, p, verdict))
            comparisons.append((condition, baseline, n, a_only, b_only, p))

    # --- what to be careful about ---
    lines += ["", "## Things to keep in mind", "",
              "- %d seeds: %s." % (len(config.SEEDS), ", ".join(str(s) for s in config.SEEDS)),
              "- Per-task numbers come from only 10 episodes each, so they are far too",
              "  noisy to quote on their own.",
              "- Every condition saw the same starting scenes, so the comparisons above",
              "  are scene by scene, which is much more sensitive than comparing averages.",
              "- Overlapping confidence intervals do NOT prove two conditions are the",
              "  same. Read the p-value, which uses the pairing.",
              ]

    summary_path = os.path.join(config.ANALYSIS_DIR, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    results_csv = os.path.join(config.ANALYSIS_DIR, "results.csv")
    with open(results_csv, "w") as f:
        f.write("condition,episodes,successes,success_rate,ci_low,ci_high\n")
        for condition in sorted(pooled):
            outcomes = list(pooled[condition].values())
            low, high = wilson_interval(sum(outcomes), len(outcomes))
            f.write("%s,%d,%d,%.2f,%.2f,%.2f\n"
                    % (condition, len(outcomes), sum(outcomes),
                       100.0 * sum(outcomes) / len(outcomes), low, high))

    comparisons_csv = os.path.join(config.ANALYSIS_DIR, "comparisons.csv")
    with open(comparisons_csv, "w") as f:
        f.write("condition_a,condition_b,paired_episodes,a_only,b_only,p_value\n")
        for row in comparisons:
            f.write("%s,%s,%d,%d,%d,%.6f\n" % row)

    print("\nwrote %s" % summary_path)
    print("      %s" % results_csv)
    print("      %s" % comparisons_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
