#!/usr/bin/env python3
"""
Validate simulator accuracy against test cases and historical data.

Usage:
    python3 analysis/validate.py                    # Full validation
    python3 analysis/validate.py --historical       # Historical races only
    python3 analysis/validate.py --test-cases       # Test cases only
    python3 analysis/validate.py --debug TEST_001   # Debug single test case
"""

import json
import glob
import os
import sys
import subprocess
from collections import defaultdict

DATA_DIR = "data/historical_races"
TEST_INPUT_DIR = "data/test_cases/inputs"
TEST_EXPECTED_DIR = "data/test_cases/expected_outputs"

# Import simulator logic directly
sys.path.insert(0, "solution")
from race_simulator import simulate_race, COMPOUND_OFFSET, DEG_RATE, TEMP_REF, TEMP_FACTOR

def validate_test_cases():
    """Validate against the 100 official test cases."""
    inputs = sorted(glob.glob(os.path.join(TEST_INPUT_DIR, "*.json")))
    correct = 0
    total = 0
    failures = []
    
    for inp_path in inputs:
        test_name = os.path.basename(inp_path).replace("input_", "").replace(".json", "")
        exp_path = os.path.join(TEST_EXPECTED_DIR, f"test_{test_name}.json")
        
        if not os.path.exists(exp_path):
            # Try alternative naming
            base = os.path.basename(inp_path)
            exp_path = os.path.join(TEST_EXPECTED_DIR, base)
        
        if not os.path.exists(exp_path):
            continue
        
        with open(inp_path) as f:
            race_input = json.load(f)
        with open(exp_path) as f:
            expected = json.load(f)
        
        result = simulate_race(race_input)
        predicted = result["finishing_positions"]
        actual = expected["finishing_positions"]
        
        total += 1
        if predicted == actual:
            correct += 1
        else:
            failures.append({
                "test": test_name,
                "predicted": predicted,
                "actual": actual,
            })
    
    acc = correct / total if total else 0
    print(f"\n=== Test Case Validation ===")
    print(f"Correct: {correct}/{total} = {acc:.1%}")
    
    if failures:
        print(f"\nFailed cases ({len(failures)}):")
        for f in failures[:10]:
            print(f"\n  {f['test']}:")
            # Find first divergence
            for i, (p, a) in enumerate(zip(f["predicted"], f["actual"])):
                if p != a:
                    print(f"    First divergence at P{i+1}: predicted {p}, actual {a}")
                    break
            print(f"    Predicted top 5: {f['predicted'][:5]}")
            print(f"    Actual top 5:    {f['actual'][:5]}")
    
    return acc, correct, total

def validate_historical(max_files=5):
    """Validate against historical race data."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    
    correct = 0
    total = 0
    compound_errors = defaultdict(int)
    
    for f in files[:max_files]:
        with open(f) as fh:
            races = json.load(fh)
            if not isinstance(races, list):
                races = [races]
        
        for race in races:
            if "finishing_positions" not in race:
                continue
            
            result = simulate_race(race)
            predicted = result["finishing_positions"]
            actual = race["finishing_positions"]
            
            total += 1
            if predicted == actual:
                correct += 1
            else:
                # Analyze what kind of error
                # Find the driver we got wrong first
                for i, (p, a) in enumerate(zip(predicted, actual)):
                    if p != a:
                        # What compounds did they use?
                        strats = race["strategies"]
                        for _, s in strats.items():
                            if s["driver_id"] == p:
                                key = f"predicted_{s['starting_tire']}_over_{a}"
                                compound_errors[key] += 1
                        break
    
    acc = correct / total if total else 0
    print(f"\n=== Historical Race Validation ===")
    print(f"Files: {min(max_files, len(files))}, Races: {total}")
    print(f"Correct: {correct}/{total} = {acc:.1%}")
    
    if compound_errors:
        print(f"\nTop error patterns:")
        for k, v in sorted(compound_errors.items(), key=lambda x: -x[1])[:10]:
            print(f"  {k}: {v}")
    
    return acc, correct, total

def debug_single_race(race_id_or_file):
    """Deep debug of a single race."""
    # Load the race
    if os.path.exists(race_id_or_file):
        with open(race_id_or_file) as f:
            race = json.load(f)
    else:
        # Search in historical data
        race = None
        for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))[:5]:
            with open(f) as fh:
                races = json.load(fh)
                if not isinstance(races, list):
                    races = [races]
                for r in races:
                    if r.get("race_id") == race_id_or_file:
                        race = r
                        break
            if race:
                break
        
        if not race:
            # Try test cases
            for f in glob.glob(os.path.join(TEST_INPUT_DIR, "*.json")):
                with open(f) as fh:
                    r = json.load(fh)
                    if r.get("race_id") == race_id_or_file:
                        race = r
                        break
    
    if not race:
        print(f"Race '{race_id_or_file}' not found")
        return
    
    config = race["race_config"]
    print(f"\n=== Debug: {race.get('race_id', '?')} ===")
    print(f"Track: {config['track']}")
    print(f"Laps: {config['total_laps']}, Base: {config['base_lap_time']}s")
    print(f"Pit time: {config['pit_lane_time']}s, Temp: {config['track_temp']}°C")
    print(f"\nCurrent params:")
    print(f"  COMPOUND_OFFSET = {COMPOUND_OFFSET}")
    print(f"  DEG_RATE = {DEG_RATE}")
    print(f"  TEMP_REF = {TEMP_REF}, TEMP_FACTOR = {TEMP_FACTOR}")
    
    # Compute all times
    from race_simulator import compute_driver_total_time, temp_multiplier
    
    temp_mult = temp_multiplier(config["track_temp"])
    print(f"  temp_mult = {temp_mult:.4f}")
    
    driver_times = {}
    for _, strat in race["strategies"].items():
        did = strat["driver_id"]
        t = compute_driver_total_time(strat, config)
        driver_times[did] = t
    
    predicted = sorted(driver_times.keys(), key=lambda d: driver_times[d])
    actual = race.get("finishing_positions", [])
    
    print(f"\n{'Pos':>4} {'Pred':>6} {'Actual':>8} {'Match':>6} {'Time':>12} {'Strategy'}")
    print("-" * 70)
    
    for i in range(min(20, len(predicted))):
        p_driver = predicted[i]
        a_driver = actual[i] if i < len(actual) else "?"
        match = "✓" if p_driver == a_driver else "✗"
        t = driver_times[p_driver]
        
        # Get strategy
        strat = None
        for _, s in race["strategies"].items():
            if s["driver_id"] == p_driver:
                strat = s
                break
        
        if strat:
            pits = strat.get("pit_stops", [])
            strat_str = f"{strat['starting_tire']}"
            if pits:
                strat_str += " -> " + " -> ".join(f"[L{p['lap']}]{p['to_tire']}" for p in pits)
        else:
            strat_str = "?"
        
        print(f"{i+1:>4} {p_driver:>6} {a_driver:>8} {match:>6} {t:>12.3f}  {strat_str}")
    
    if actual:
        # Find where it goes wrong
        for i, (p, a) in enumerate(zip(predicted, actual)):
            if p != a:
                print(f"\nFirst error at P{i+1}: predicted {p} but actual {a}")
                
                # Time difference between these two
                if p in driver_times and a in driver_times:
                    diff = driver_times[p] - driver_times[a]
                    print(f"  {p} time: {driver_times[p]:.4f}s")
                    print(f"  {a} time: {driver_times[a]:.4f}s")
                    print(f"  Difference: {diff:+.4f}s")
                    print(f"  Our model predicts {p} is FASTER, but actually {a} finished first")
                    print(f"  This suggests {p}'s time is overestimated or {a}'s is underestimated")
                break

def analyze_parameter_sensitivity(races_sample):
    """Show how accuracy changes as we vary each parameter."""
    from race_simulator import simulate_race
    
    base_params = {
        "COMPOUND_OFFSET": dict(COMPOUND_OFFSET),
        "DEG_RATE": dict(DEG_RATE),
        "TEMP_REF": TEMP_REF,
        "TEMP_FACTOR": TEMP_FACTOR,
    }
    
    # This would need to monkey-patch the simulator module
    # For now, just print current performance
    correct = sum(
        1 for r in races_sample
        if "finishing_positions" in r
        and simulate_race(r)["finishing_positions"] == r["finishing_positions"]
    )
    print(f"\nCurrent accuracy on sample: {correct}/{len(races_sample)} = {correct/len(races_sample):.1%}")

def main():
    args = sys.argv[1:]
    
    print("=" * 60)
    print("F1 Simulator Validation")
    print("=" * 60)
    print(f"\nCurrent Parameters:")
    print(f"  COMPOUND_OFFSET = {COMPOUND_OFFSET}")
    print(f"  DEG_RATE = {DEG_RATE}")
    print(f"  TEMP_REF = {TEMP_REF}")
    print(f"  TEMP_FACTOR = {TEMP_FACTOR}")
    
    if "--debug" in args:
        idx = args.index("--debug")
        race_id = args[idx + 1] if idx + 1 < len(args) else None
        if race_id:
            debug_single_race(race_id)
        return
    
    if "--test-cases" in args or not args:
        if os.path.exists(TEST_INPUT_DIR):
            validate_test_cases()
        else:
            print(f"\nTest cases directory not found: {TEST_INPUT_DIR}")
    
    if "--historical" in args or not args:
        if os.path.exists(DATA_DIR):
            validate_historical(max_files=10)
        else:
            print(f"\nHistorical data directory not found: {DATA_DIR}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
