import argparse
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from math_equiv import extract_final_answer_from_completion, is_equiv


def pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return float("nan")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, default="results/math500/raw")
    ap.add_argument("--run_contains", type=str, default="MATH500")
    ap.add_argument("--out_dir", type=str, default="results/math500/eval")
    ap.add_argument("--max_k", type=int, default=8)
    args = ap.parse_args()

    folder = Path(args.folder)
    files = sorted([f for f in folder.glob("*.csv") if args.run_contains in f.name])
    if not files:
        raise SystemExit(f"No CSVs found in {folder} containing '{args.run_contains}'")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # recompute robust correctness
    df["pred_eval"] = df["completion"].astype(str).apply(extract_final_answer_from_completion)
    df["correct_eval"] = [
        int(is_equiv(p, g))
        for p, g in zip(df["pred_eval"].astype(str), df["gold"].astype(str))
    ]

    ks = list(range(1, args.max_k + 1))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()

    for method, d in df.groupby("method"):
        grouped = d.groupby("example_id")["correct_eval"].apply(list)

        passk_vals = {}
        for k in ks:
            vals = []
            for corr_list in grouped:
                n = len(corr_list)
                c = sum(int(x) for x in corr_list)
                v = pass_at_k(n, c, k)
                if not math.isnan(v):
                    vals.append(v)
            if len(vals) == 0:
                passk_vals[k] = float("nan")   # <- instead of 0
            else:
                passk_vals[k] = float(sum(vals) / len(vals))
                
            #passk_vals[k] = float(sum(vals) / max(len(vals), 1))
            
        xs = [k for k,v in passk_vals.items() if not math.isnan(v)]
        ys = [v for k,v in passk_vals.items() if not math.isnan(v)]
        plt.plot(xs, ys, marker="o", label=method)
        #plt.plot(list(passk_vals.keys()), list(passk_vals.values()), marker="o", label=method)

    plt.xlabel("k")
    plt.ylabel("pass@k Accuracy")
    plt.title("pass@k on MATH500 - Qwen/Qwen2.5-Math-1.5B-Instruct")
    plt.grid(True)
    plt.legend()

    fig_path = out_dir / "passk.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"[OK] pass@k plot -> {fig_path}")


if __name__ == "__main__":
    main()
