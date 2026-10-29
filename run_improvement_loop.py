"""
Single-command runner for the Contract Intelligence Assistant Self-Improvement Loop.
Usage:
    python run_improvement_loop.py
"""
import sys
import time
from src.self_improvement_loop import SelfImprovementLoop

def main():
    start_time = time.time()
    loop = SelfImprovementLoop()
    result = loop.run_full_self_improvement_suite()
    total_time = time.time() - start_time
    print(f"\nAll iterations completed successfully in {total_time:.2f} seconds.")
    print(f"Log persisted to experiment_log.json")
    print(f"Configs persisted to configs/\n")

if __name__ == "__main__":
    main()
