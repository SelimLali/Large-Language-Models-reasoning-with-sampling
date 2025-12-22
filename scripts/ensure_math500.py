import json
from pathlib import Path

def ensure_math500(data_dir: Path) -> Path:
    """
    Downloads HuggingFaceH4/MATH-500 test split (500 rows) and caches into data_dir/test.jsonl.
    Dataset fields: problem, solution, answer, subject, level, unique_id.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "test.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[OK] Using cached dataset: {out_path}")
        return out_path

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Please `pip install datasets`") from e

    print("[DL] Downloading HuggingFaceH4/MATH-500 (split=test)...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[OK] Saved: {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
    return out_path

if __name__ == "__main__":
    ensure_math500(Path("data") / "math500")
