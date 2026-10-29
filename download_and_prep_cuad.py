import os
import json
import random
import hashlib
from pathlib import Path

DATA_DIR = Path("data")
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
CORPUS_DIR = DATA_DIR / "corpus" / "cuad"

def mini(tests, n=194):
    if len(tests) <= n:
        return tests
    def key(t):
        random.seed(t["snippets"][0]["file_path"])
        return random.random()
    return sorted(tests, key=key)[:n]

def assign_split(query: str) -> str:
    h = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if h % 10 < 3 else "train"

def main():
    cuad_bench_file = DATA_DIR / "benchmarks" / "cuad.json"
    raw = cuad_bench_file.read_text(encoding="utf-8")
    data = json.loads(raw)
    all_tests = data["tests"]
    print(f"Loaded full CUAD benchmark: {len(all_tests)} total queries.")

    # Select 194 CUAD mini queries
    cuad_mini = mini(all_tests, n=194)
    print(f"Selected CUAD mini: {len(cuad_mini)} queries.")

    # Check unique files
    unique_contracts = set()
    for t in cuad_mini:
        for s in t["snippets"]:
            unique_contracts.add(s["file_path"])
    print(f"Unique contracts required: {len(unique_contracts)}")

    # Verify character offsets: document[start:end] == snippet['answer']
    print("\nVerifying snippet offsets: document[span[0]:span[1]] == snippet['answer']...")
    verified_count = 0
    mismatch_count = 0
    missing_count = 0

    for i, t in enumerate(cuad_mini):
        for s_idx, s in enumerate(t["snippets"]):
            fp = DATA_DIR / s["file_path"]
            if not fp.exists():
                fp = DATA_DIR / "corpus" / s["file_path"]
            if not fp.exists():
                missing_count += 1
                continue
            doc_text = fp.read_text(encoding="utf-8")
            start, end = s["span"]
            expected = s.get("answer", "")
            actual = doc_text[start:end]
            if actual == expected:
                verified_count += 1
            else:
                mismatch_count += 1
                if mismatch_count <= 3:
                    print(f"  [Mismatch sample {mismatch_count}] Query {i} snippet {s_idx} in {s['file_path']}:")
                    print(f"    Expected: {repr(expected[:60])}")
                    print(f"    Actual  : {repr(actual[:60])}")

    print(f"Verification complete: {verified_count} exact matches, {mismatch_count} mismatches, {missing_count} missing files.")

    # Train / test split analysis
    splits = {"train": 0, "test": 0}
    for i, t in enumerate(cuad_mini):
        sp = assign_split(t["query"])
        t["split"] = sp
        t["id"] = f"cuad_mini_{i:03d}"
        splits[sp] += 1

    print(f"\nCUAD mini train/test split: train={splits['train']}, test={splits['test']} (total={len(cuad_mini)})")

    # Save prepared dataset
    out_file = DATA_DIR / "cuad_mini.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "name": "LegalBench-RAG-mini (CUAD)",
            "total_queries": len(cuad_mini),
            "train_queries": splits["train"],
            "test_queries": splits["test"],
            "unique_contracts": len(unique_contracts),
            "queries": cuad_mini
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved prepared mini dataset with splits to {out_file}")

if __name__ == "__main__":
    main()
