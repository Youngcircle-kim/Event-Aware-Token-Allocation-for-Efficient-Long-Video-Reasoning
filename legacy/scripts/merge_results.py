"""
Merge results from separate runs (e.g., uniform_only/ + event_aware_only/)
into a single comparison table.

Usage:
    python scripts/merge_results.py \
        --result_dirs ./results/uniform_only ./results/event_aware_only \
        --budgets 8 16 32 \
        --output ./results/merged
"""
import argparse
import json
from pathlib import Path


def load_per_method_curves(result_dirs, budgets):
    """Find all per-method results across the given dirs."""
    curves = {}  # {method_name: {budget: accuracy}}
    
    for d in result_dirs:
        d = Path(d)
        for budget in budgets:
            for report_file in d.glob(f"*_T{budget}_report.json"):
                # Filename: "uniform_T16_report.json" → method = "uniform"
                method_name = report_file.stem.replace(f"_T{budget}_report", "")
                
                with open(report_file) as f:
                    report = json.load(f)
                
                if method_name not in curves:
                    curves[method_name] = {}
                curves[method_name][budget] = report["overall_accuracy"]
    
    return curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dirs", nargs="+", required=True,
                        help="Directories containing run results")
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--output", default="./results/merged")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    curves = load_per_method_curves(args.result_dirs, args.budgets)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("MERGED ACCURACY-BUDGET CURVE")
    print("=" * 70)
    print(f"\n{'Method':<20s}" + "".join(f"T={b:<8d}" for b in args.budgets))
    print("-" * 70)
    for method, by_budget in curves.items():
        row = f"{method:<20s}"
        for b in args.budgets:
            v = by_budget.get(b, None)
            row += f"{v:<10.3f}" if v is not None else f"{'--':<10s}"
        print(row)
    
    # Save merged JSON
    save_path = output_dir / "accuracy_budget_curve.json"
    with open(save_path, "w") as f:
        json.dump(curves, f, indent=2)
    print(f"\nMerged curve saved to {save_path}")


if __name__ == "__main__":
    main()