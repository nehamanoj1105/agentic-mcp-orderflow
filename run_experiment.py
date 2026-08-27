"""
CLI Research Experiment Runner.

Runs specified experiments (Exp 1 through Exp 6) or executes the entire research suite.
"""

import argparse
import sys
from experiments import (
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
    run_experiment_4,
    run_experiment_5,
    run_experiment_6
)


def main():
    parser = argparse.ArgumentParser(description="Run Microstructure ML Research Experiments")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4, 5, 6], help="Specific experiment number to run (1-6)")
    parser.add_argument("--all", action="store_true", help="Run all 6 research experiments sequentially")
    args = parser.parse_args()

    if args.all or args.exp is None:
        print("Running full research experiment suite (Experiments 1 - 6)...")
        run_experiment_1()
        run_experiment_2()
        run_experiment_3()
        run_experiment_4()
        run_experiment_5()
        run_experiment_6()
        print("\nAll 6 research experiments completed successfully!")
    else:
        exp_map = {
            1: run_experiment_1,
            2: run_experiment_2,
            3: run_experiment_3,
            4: run_experiment_4,
            5: run_experiment_5,
            6: run_experiment_6
        }
        exp_map[args.exp]()


if __name__ == "__main__":
    main()
