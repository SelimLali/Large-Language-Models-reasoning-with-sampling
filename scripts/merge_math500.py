import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="results/math500/raw")
    ap.add_argument("--output_csv", type=str, default="results/math500/merged/merged.csv")
    ap.add_argument("--run_prefix", type=str, default="", help="Optional filter: only merge files containing this substring")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.csv"))
    if args.run_prefix:
        files = [f for f in files if args.run_prefix in f.name]

    if not files:
        raise SystemExit(f"No CSVs found in {in_dir} (prefix='{args.run_prefix}')")

    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)

    df.to_csv(out_path, index=False)
    print(f"[OK] Merged {len(files)} files, {len(df)} rows -> {out_path}")

if __name__ == "__main__":
    main()
