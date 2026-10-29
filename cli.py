"""
Command Line Interface (CLI) for Contract Intelligence Assistant.
Supports:
  1. Single query search (python cli.py search "query")
  2. Interactive shell (python cli.py interactive)
  3. Evaluation runner (python cli.py evaluate --split train/test)
  4. Self-improvement loop runner (python cli.py loop)
"""
import sys
import argparse
import json
from pathlib import Path

from src.config import PipelineConfig
from src.pipeline import ContractIntelligenceEngine
from src.evaluator import Evaluator
from src.self_improvement_loop import SelfImprovementLoop

def cmd_search(args):
    config = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    engine = ContractIntelligenceEngine(config=config)
    
    print(f"\nSearching across indexed contracts...")
    print(f"Query: \"{args.query}\"")
    if args.contract:
        print(f"Filter: {args.contract}")
    
    resp = engine.search(query=args.query, top_k=args.top_k, contract_filter=args.contract)
    print(f"Search completed in {resp.latency_ms:.2f} ms")
    print("=" * 60)

    if resp.is_abstained:
        print("\n[!] SYSTEM ABSTENTION")
        print(f"Reason: {resp.abstention_reason}")
        print(f"Confidence score: {resp.confidence:.4f}")
        print("No matching clause found in the indexed agreements.\n")
    else:
        print(f"\nFound {len(resp.passages)} candidate clause(s):\n")
        for p in resp.passages:
            print(f"--- [Rank {p.rank}] Score: {p.score:.4f} ---")
            print(f"Document : {p.file_path}")
            print(f"Offsets  : [{p.start_char}:{p.end_char}] ({p.end_char - p.start_char} characters)")
            print(f"Clause   :")
            print(f"{p.text.strip()}")
            print("-" * 60)
        print()

def cmd_interactive(args):
    config = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    engine = ContractIntelligenceEngine(config=config)
    print("\n" + "=" * 60)
    print("Contract Intelligence Assistant - Interactive Mode")
    print("Type your natural language contract question below.")
    print("Type 'exit' or 'quit' to terminate.")
    print("=" * 60 + "\n")

    while True:
        try:
            q = input("\nEnter query > ").strip()
            if not q:
                continue
            if q.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive mode.")
                break

            resp = engine.search(query=q, top_k=args.top_k)
            print(f"\nLatency: {resp.latency_ms:.1f}ms | Confidence: {resp.confidence:.4f}")

            if resp.is_abstained:
                print(f">> [ABSTAINED] {resp.abstention_reason}")
            else:
                print(f">> Top match in {resp.passages[0].file_path} [{resp.passages[0].start_char}:{resp.passages[0].end_char}]:")
                print(f"   {resp.passages[0].text[:300]}...")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

def cmd_evaluate(args):
    config = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    engine = ContractIntelligenceEngine(config=config)
    evaluator = Evaluator(data_dir=config.data_dir)
    
    print(f"\nRunning evaluation on {args.split.upper()} split...")
    report = evaluator.run_evaluation(engine=engine, split=args.split)
    report.print_summary()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Detailed evaluation metrics saved to {args.output}")

def cmd_loop(args):
    loop = SelfImprovementLoop()
    loop.run_full_self_improvement_suite()

def main():
    parser = argparse.ArgumentParser(description="Contract Intelligence Assistant CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search command
    p_search = subparsers.add_parser("search", help="Execute single query search")
    p_search.add_argument("query", type=str, help="Natural language contract question")
    p_search.add_argument("--config", type=str, default=None, help="Path to config JSON")
    p_search.add_argument("--top_k", type=int, default=5, help="Number of passages to return")
    p_search.add_argument("--contract", type=str, default=None, help="Optional contract filter")
    p_search.set_defaults(func=cmd_search)

    # Interactive command
    p_inter = subparsers.add_parser("interactive", help="Start interactive search prompt")
    p_inter.add_argument("--config", type=str, default=None, help="Path to config JSON")
    p_inter.add_argument("--top_k", type=int, default=3, help="Number of passages to return")
    p_inter.set_defaults(func=cmd_interactive)

    # Evaluate command
    p_eval = subparsers.add_parser("evaluate", help="Run benchmark evaluation")
    p_eval.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split")
    p_eval.add_argument("--config", type=str, default=None, help="Path to config JSON")
    p_eval.add_argument("--output", type=str, default=None, help="Path to save output JSON")
    p_eval.set_defaults(func=cmd_evaluate)

    # Improvement loop command
    p_loop = subparsers.add_parser("loop", help="Run the automated self-improvement loop")
    p_loop.set_defaults(func=cmd_loop)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)

if __name__ == "__main__":
    main()
