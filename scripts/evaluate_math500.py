import argparse
import json
from pathlib import Path

import pandas as pd
from math_equiv import extract_final_answer_from_completion, is_equiv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/math500/eval")
    ap.add_argument("--write_eval_csv", action="store_true", help="Save a CSV with pred_eval + correct_eval columns")
    args = ap.parse_args()

    df = pd.read_csv(args.merged_csv)

    # recompute prediction + correctness from completion
    df["pred_eval"] = df["completion"].astype(str).apply(extract_final_answer_from_completion)
    df["correct_eval"] = [
        int(is_equiv(p, g))
        for p, g in zip(df["pred_eval"].astype(str), df["gold"].astype(str))
    ]

    acc = float(df["correct_eval"].mean()) if len(df) else 0.0
    by_method = df.groupby("method")["correct_eval"].mean().to_dict() if "method" in df.columns else {}
    by_seed = df.groupby("seed")["correct_eval"].mean().to_dict() if "seed" in df.columns else {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = {
        "n": int(len(df)),
        "accuracy": acc,
        "accuracy_by_method": {k: float(v) for k, v in by_method.items()},
        "accuracy_by_seed": {str(k): float(v) for k, v in by_seed.items()},
    }
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(f"[OK] scores.json -> {out_dir/'scores.json'}")

    # per-example jsonl
    per_ex_path = out_dir / "per_example.jsonl"
    with per_ex_path.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps({
                "example_id": r.get("example_id", ""),
                "seed": int(r.get("seed", -1)),
                "method": r.get("method", ""),
                "correct": int(r.get("correct_eval", 0)),
                "gold": r.get("gold", ""),
                "pred": r.get("pred_eval", ""),
            }, ensure_ascii=False) + "\n")
    print(f"[OK] per_example.jsonl -> {per_ex_path}")

    if args.write_eval_csv:
        out_csv = out_dir / "merged_with_eval.csv"
        df.to_csv(out_csv, index=False)
        print(f"[OK] merged_with_eval.csv -> {out_csv}")
    
    # Bar plot (MATH500)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Accuracies in %
    acc_by_method = scores.get("accuracy_by_method", {})
    labels = ["base", "lowtemp", "power_mh"]
    values = [100.0 * float(acc_by_method.get(k, 0.0)) for k in labels]
    colors = ["#9ecae1", "#3182bd", "#08519c"]
    
    # Plot
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, width=0.4, color=colors)

    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)

    # Title with number of evaluated rows
    #n_eval = scores.get("n", len(df))
    n_eval = int(df["example_id"].nunique()) if "example_id" in df.columns else scores.get("n", len(df))
    plt.title(f'Reasoning on "{n_eval}" examples of MATH500 dataset')

    #values on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    
    # Legend (one entry per method/color)
    legend_handles = [
        mpatches.Patch(color=colors[0], label="Base Qwen/Qwen2.5-Math-1.5B-Instruct"),
        mpatches.Patch(color=colors[1], label="Low-temp"),
        mpatches.Patch(color=colors[2], label="Training free Power-MH"),
    ]
    #plt.legend(handles=legend_handles, loc="upper right", frameon=True)
    
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    # Save figure
    fig_path = out_dir / "math500_barplot.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] bar plot -> {fig_path}")


if __name__ == "__main__":
    main()
