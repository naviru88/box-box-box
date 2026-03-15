#!/usr/bin/env python3
"""
QUICK START: Run this immediately after getting access to the data.

python3 analysis/quick_start.py

This script:
1. Loads just a few races
2. Prints the FULL structure so we know EXACTLY what fields are available
3. If lap_times are present, extracts params directly
4. If not, does a fast grid search
5. Prints accuracy and next steps
"""

import json
import glob
import os
import sys
import math
from collections import defaultdict

DATA_DIR = "data/historical_races"
TEST_INPUT_DIR = "data/test_cases/inputs"
TEST_EXPECTED_DIR = "data/test_cases/expected_outputs"

def load_first_few():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    if not files:
        print(f"ERROR: No files found in {DATA_DIR}")
        sys.exit(1)
    
    with open(files[0]) as f:
        data = json.load(f)
    
    if isinstance(data, list):
        races = data
    else:
        races = [data]
    
    return races, files

def print_structure(race):
    """Print EVERYTHING about a race to understand the full schema."""
    print("\n" + "="*60)
    print("FULL RACE STRUCTURE")
    print("="*60)
    
    def print_dict(d, indent=0, max_items=3):
        prefix = "  " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}: {{")
                # Only show first max_items keys of nested dict
                items = list(v.items())
                for kk, vv in items[:max_items]:
                    if isinstance(vv, (dict, list)):
                        print(f"{prefix}  {kk}: {type(vv).__name__}({len(vv)})")
                    else:
                        print(f"{prefix}  {kk}: {repr(vv)}")
                if len(items) > max_items:
                    print(f"{prefix}  ... ({len(items)-max_items} more)")
                print(f"{prefix}}}")
            elif isinstance(v, list):
                print(f"{prefix}{k}: [{type(v[0]).__name__ if v else 'empty'} × {len(v)}]")
                if v and isinstance(v[0], (int, float)):
                    print(f"{prefix}  first 5: {v[:5]}")
            else:
                print(f"{prefix}{k}: {repr(v)}")
    
    print_dict(race, max_items=5)
    
    # Print first strategy fully
    print("\nFIRST STRATEGY (pos1):")
    print(json.dumps(race["strategies"]["pos1"], indent=2))
    
    # List ALL keys
    print(f"\nALL RACE KEYS: {sorted(race.keys())}")
    print(f"ALL CONFIG KEYS: {sorted(race['race_config'].keys())}")
    
    if "finishing_positions" in race:
        print(f"\nfinishing_positions present: {race['finishing_positions'][:5]}...")
    
    # Check for extra time data
    for key in ["lap_times", "driver_lap_times", "lap_data", "total_times", 
                "sector_times", "timing_data", "results"]:
        if key in race:
            print(f"\n*** FOUND {key}! ***")
            val = race[key]
            if isinstance(val, dict):
                first_key = list(val.keys())[0]
                first_val = val[first_key]
                print(f"  Keys: {list(val.keys())[:3]}...")
                print(f"  First entry ({first_key}): {first_val[:5] if isinstance(first_val, list) else first_val}")

def analyze_lap_times_if_available(races):
    """Extract exact params if raw lap times exist."""
    for r in races:
        for key in ["lap_times", "driver_lap_times", "lap_data"]:
            if key in r:
                print(f"\n🎯 Raw lap times found under '{key}'!")
                return extract_exact_params(races, key)
    
    print("\n⚠️  No raw lap times found - will use ranking-based optimization")
    return None

def extract_exact_params(races, lap_key):
    """Linear regression on raw lap times to get exact parameters."""
    from collections import defaultdict
    
    # Collect: (tire_age, compound, lap_time - base, track_temp)
    obs = defaultdict(list)  # compound -> [(age, adjusted_time, temp)]
    
    for race in races:
        if lap_key not in race:
            continue
        
        config = race["race_config"]
        base = config["base_lap_time"]
        temp = config["track_temp"]
        lap_times_by_driver = race[lap_key]
        
        for _, strat in race["strategies"].items():
            did = strat["driver_id"]
            if did not in lap_times_by_driver:
                continue
            
            laps = lap_times_by_driver[did]
            pit_sched = {p["lap"]: p["to_tire"] for p in strat.get("pit_stops", [])}
            
            tire = strat["starting_tire"]
            age = 0
            
            for lap_num, lap_time in enumerate(laps, 1):
                age += 1   # increments BEFORE lap time (regulations)
                adjusted = lap_time - base
                obs[tire].append((age, adjusted, temp))
                if lap_num in pit_sched:
                    tire = pit_sched[lap_num]
                    age = 0
    
    params = {}
    print("\nDirect regression results:")
    print(f"{'Compound':>10} {'Offset':>10} {'Deg/lap':>10} {'N obs':>8}")
    print("-" * 42)
    
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        data = obs[compound]
        if len(data) < 2:
            continue
        
        n = len(data)
        # Fit: adjusted = offset + deg * age (ignoring temp first pass)
        ages = [d[0] for d in data]
        vals = [d[1] for d in data]
        
        mean_x = sum(ages) / n
        mean_y = sum(vals) / n
        
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(ages, vals)) / n
        var = sum((x - mean_x)**2 for x in ages) / n
        
        deg = cov / var if var > 0 else 0
        offset = mean_y - deg * mean_x
        
        params[compound] = {"offset": offset, "deg_rate": deg}
        print(f"{compound:>10} {offset:>10.4f} {deg:>10.6f} {n:>8}")
    
    # Now fit temperature effect
    if "SOFT" in params and "MEDIUM" in params:
        # Residuals after fitting offset + deg
        residuals_by_temp = defaultdict(list)
        for compound, data in obs.items():
            if compound not in params:
                continue
            p = params[compound]
            for age, val, temp in data:
                predicted = p["offset"] + p["deg_rate"] * age
                residual = val - predicted
                residuals_by_temp[temp].append(residual)
        
        # Fit: residual = alpha * (temp - temp_ref)
        # Try different temp_refs
        best_temp_ref = 30.0
        best_temp_fac = 0.0
        # TODO: fit this properly
        
        print(f"\nTemperature analysis:")
        for temp in sorted(residuals_by_temp.keys()):
            r = residuals_by_temp[temp]
            avg = sum(r) / len(r)
            print(f"  {temp}°C: avg residual = {avg:.4f}s ({len(r)} laps)")
    
    # Convert MEDIUM to reference (offset=0)
    med_offset = params.get("MEDIUM", {}).get("offset", 0.0)
    
    result = {
        "soft_offset": params.get("SOFT", {}).get("offset", -1.5) - med_offset,
        "medium_offset": 0.0,
        "hard_offset": params.get("HARD", {}).get("offset", 0.8) - med_offset,
        "soft_deg": params.get("SOFT", {}).get("deg_rate", 0.10),
        "medium_deg": params.get("MEDIUM", {}).get("deg_rate", 0.05),
        "hard_deg": params.get("HARD", {}).get("deg_rate", 0.02),
        "temp_ref": 30.0,
        "temp_factor": 0.01,
    }
    
    print(f"\nExtracted parameters:")
    for k, v in result.items():
        print(f"  {k}: {v:.6f}")
    
    return result

def fast_grid_search(races):
    """Fast grid search with essential param ranges."""
    sample = [r for r in races[:100] if "finishing_positions" in r]
    
    if not sample:
        print("No races with finishing positions found!")
        return None
    
    def sim_time(strat, config, soft_off, hard_off, soft_deg, med_deg, hard_deg, temp_ref, temp_fac):
        base = config["base_lap_time"]
        total_laps = config["total_laps"]
        pit_time = config["pit_lane_time"]
        track_temp = config["track_temp"]
        temp_mult = 1.0 + (track_temp - temp_ref) * temp_fac
        
        offsets = {"SOFT": soft_off, "MEDIUM": 0.0, "HARD": hard_off}
        degs = {"SOFT": soft_deg, "MEDIUM": med_deg, "HARD": hard_deg}
        
        pit_sched = {p["lap"]: p["to_tire"] for p in strat.get("pit_stops", [])}
        tire = strat["starting_tire"]
        age = 0
        t = 0.0
        for lap in range(1, total_laps + 1):
            age += 1   # increments BEFORE lap time (regulations)
            t += base + offsets[tire] + age * degs[tire] * temp_mult
            if lap in pit_sched:
                t += pit_time
                tire = pit_sched[lap]
                age = 0
        return t
    
    def eval_params(so, ho, sd, md, hd, tr, tf):
        correct = 0
        for race in sample:
            config = race["race_config"]
            times = {}
            for _, strat in race["strategies"].items():
                did = strat["driver_id"]
                times[did] = sim_time(strat, config, so, ho, sd, md, hd, tr, tf)
            pred = sorted(times.keys(), key=lambda d: times[d])
            if pred == race["finishing_positions"]:
                correct += 1
        return correct / len(sample)
    
    best_acc = 0
    best_p = None
    
    # Key ranges to test
    for so in [-3.0, -2.0, -1.5, -1.0, -0.5]:
        for ho in [0.3, 0.5, 0.8, 1.0, 1.5]:
            for sd in [0.05, 0.08, 0.10, 0.15, 0.20]:
                for md in [0.02, 0.04, 0.05, 0.06]:
                    for hd in [0.01, 0.02, 0.03]:
                        if not (sd > md > hd):
                            continue
                        for tr in [25.0, 30.0, 35.0]:
                            for tf in [0.0, 0.01, 0.02]:
                                acc = eval_params(so, ho, sd, md, hd, tr, tf)
                                if acc > best_acc:
                                    best_acc = acc
                                    best_p = (so, ho, sd, md, hd, tr, tf)
                                    print(f"  New best: {acc:.3f} - so={so}, ho={ho}, sd={sd}, md={md}, hd={hd}, tr={tr}, tf={tf}")
    
    print(f"\nGrid search best: {best_acc:.3f}")
    print(f"Params: {best_p}")
    
    return {
        "soft_offset": best_p[0],
        "medium_offset": 0.0,
        "hard_offset": best_p[1],
        "soft_deg": best_p[2],
        "medium_deg": best_p[3],
        "hard_deg": best_p[4],
        "temp_ref": best_p[5],
        "temp_factor": best_p[6],
    }

def update_and_test(params):
    """Update simulator.py and run a quick test."""
    # Update parameters in simulator.py
    sim_path = "solution/simulator.py"
    with open(sim_path) as f:
        content = f.read()
    
    replacements = {
        '"SOFT":': f'"SOFT":   {params["soft_offset"]:.6f},',
        '"MEDIUM":': f'"MEDIUM":  {params["medium_offset"]:.6f},',
        '"HARD":': f'"HARD":    {params["hard_offset"]:.6f},',
    }
    
    # More targeted replacement using specific section markers
    # Find COMPOUND_OFFSET section and replace
    import re
    
    # Replace SOFT in COMPOUND_OFFSET
    content = re.sub(
        r'(COMPOUND_OFFSET\s*=\s*\{[^}]*"SOFT":\s*)-?\d+\.\d+',
        f'\\g<1>{params["soft_offset"]:.6f}',
        content
    )
    content = re.sub(
        r'(COMPOUND_OFFSET\s*=\s*\{[^}]*"HARD":\s*)-?\d+\.\d+',
        f'\\g<1>{params["hard_offset"]:.6f}',
        content
    )
    
    # Replace DEG_RATE values
    content = re.sub(
        r'(DEG_RATE\s*=\s*\{[^}]*"SOFT":\s*)\d+\.\d+',
        f'\\g<1>{params["soft_deg"]:.6f}',
        content
    )
    content = re.sub(
        r'(DEG_RATE\s*=\s*\{[^}]*"MEDIUM":\s*)\d+\.\d+',
        f'\\g<1>{params["medium_deg"]:.6f}',
        content
    )
    content = re.sub(
        r'(DEG_RATE\s*=\s*\{[^}]*"HARD":\s*)\d+\.\d+',
        f'\\g<1>{params["hard_deg"]:.6f}',
        content
    )
    
    # TEMP_REF and TEMP_FACTOR
    content = re.sub(r'TEMP_REF\s*=\s*\d+\.\d+', f'TEMP_REF = {params["temp_ref"]:.1f}', content)
    content = re.sub(r'TEMP_FACTOR\s*=\s*\d+\.\d+', f'TEMP_FACTOR = {params["temp_factor"]:.6f}', content)
    
    with open(sim_path, "w") as f:
        f.write(content)
    
    print(f"\nUpdated {sim_path}")
    
    # Save params json
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/optimized_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print("Saved analysis/optimized_params.json")

def run_test_cases():
    """Quick check on test cases."""
    inputs = sorted(glob.glob(os.path.join(TEST_INPUT_DIR, "*.json")))
    if not inputs:
        print(f"\nNo test cases found in {TEST_INPUT_DIR}")
        return
    
    sys.path.insert(0, "solution")
    # Re-import to get updated params
    import importlib
    import race_simulator as simulator
    importlib.reload(race_simulator)
    
    correct = 0
    total = 0
    
    for inp_path in inputs:
        base = os.path.basename(inp_path)
        # Try to find expected output
        for exp_dir in [TEST_EXPECTED_DIR]:
            exp_path = os.path.join(exp_dir, base)
            if not os.path.exists(exp_path):
                # Try with different prefix
                exp_path = os.path.join(exp_dir, base.replace("test_", "").replace("input_", ""))
            if os.path.exists(exp_path):
                with open(inp_path) as f:
                    race = json.load(f)
                with open(exp_path) as f:
                    expected = json.load(f)
                
                result = simulator.simulate_race(race)
                if result["finishing_positions"] == expected["finishing_positions"]:
                    correct += 1
                total += 1
                break
    
    if total > 0:
        print(f"\nTest case accuracy: {correct}/{total} = {correct/total:.1%}")
    else:
        print("\nNo matching test case pairs found")

def main():
    print("=" * 60)
    print("F1 Simulator - Quick Start")
    print("=" * 60)
    
    # Load data
    races, files = load_first_few()
    print(f"\nFound {len(files)} data files")
    print(f"First file: {files[0]} ({len(races)} races)")
    
    # Print structure
    print_structure(races[0])
    
    # Check for raw lap times
    params = analyze_lap_times_if_available(races)
    
    if params is None:
        # Need to do optimization
        print("\n--- Running fast grid search ---")
        # Load more data for better accuracy
        all_races = []
        for f in files[:3]:
            with open(f) as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    all_races.extend(data)
                else:
                    all_races.append(data)
        
        params = fast_grid_search(all_races)
    
    if params:
        update_and_test(params)
        
        # Quick test
        run_test_cases()
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("1. Run: python3 analysis/calibrate.py  (full optimization)")
        print("2. Run: python3 analysis/validate.py   (check accuracy)")
        print("3. Run: ./test_runner.sh                (official test)")
        print("="*60)

if __name__ == "__main__":
    main()
