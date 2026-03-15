#!/usr/bin/env python3
"""
Calibrate simulator parameters from historical race data.

This script:
1. Loads historical race data
2. Runs optimization to find best tire model parameters
3. Updates solution/simulator.py with the optimized values

Run this ONCE after getting the data:
    python3 analysis/calibrate.py

Then test:
    ./test_runner.sh
"""

import json
import glob
import os
import sys
import re
import math
from itertools import product as iproduct

DATA_DIR = "data/historical_races"
SIMULATOR_PATH = "solution/simulator.py"
PARAMS_PATH = "analysis/optimized_params.json"

# ---- Model ----

def sim_driver(strategy, config, p):
    """Compute total race time for one driver given params array p."""
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    track_temp = config["track_temp"]
    
    soft_off, hard_off, soft_deg, med_deg, hard_deg, temp_ref, temp_fac = p
    
    offsets = {"SOFT": soft_off, "MEDIUM": 0.0, "HARD": hard_off}
    degs    = {"SOFT": soft_deg, "MEDIUM": med_deg, "HARD": hard_deg}
    
    temp_mult = 1.0 + (track_temp - temp_ref) * temp_fac
    
    pit_sched = {pit["lap"]: pit["to_tire"] for pit in strategy.get("pit_stops", [])}
    
    tire = strategy["starting_tire"]
    age = 0
    t = 0.0
    
    for lap in range(1, total_laps + 1):
        age += 1   # increments BEFORE lap time (regulations: "age increments at start of each lap")
        t += base + offsets[tire] + age * degs[tire] * temp_mult
        if lap in pit_sched:
            t += pit_time
            tire = pit_sched[lap]
            age = 0   # reset; next lap will increment to 1 before use
    
    return t

def predict_order(race, p):
    """Return predicted finishing order for a race."""
    config = race["race_config"]
    times = {}
    for _, strat in race["strategies"].items():
        did = strat["driver_id"]
        times[did] = sim_driver(strat, config, p)
    return sorted(times.keys(), key=lambda d: times[d])

def accuracy(races, p):
    """Fraction of races with exact correct order prediction."""
    correct = sum(
        1 for r in races
        if "finishing_positions" in r
        and predict_order(r, p) == r["finishing_positions"]
    )
    total = sum(1 for r in races if "finishing_positions" in r)
    return correct / total if total else 0.0

def pairwise_loss(races, p, max_races=500):
    """Count pairwise ordering violations (smoother than 0/1 loss)."""
    loss = 0
    for race in races[:max_races]:
        if "finishing_positions" not in race:
            continue
        config = race["race_config"]
        actual = race["finishing_positions"]
        times = {}
        for _, strat in race["strategies"].items():
            did = strat["driver_id"]
            times[did] = sim_driver(strat, config, p)
        
        # For each consecutive pair in actual order: winner should have lower time
        for i in range(len(actual) - 1):
            di, dj = actual[i], actual[i + 1]
            if di in times and dj in times:
                # di should be faster (less time)
                margin = times[dj] - times[di]
                if margin <= 0:
                    loss += 1 + abs(margin)
    return loss

# ---- Data Loading ----

def load_races(max_files=10):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    races = []
    for f in files[:max_files]:
        with open(f) as fh:
            data = json.load(fh)
            if isinstance(data, list):
                races.extend(data)
            else:
                races.append(data)
    return races

# ---- Optimization ----

def coarse_grid_search(races, verbose=True):
    """
    Exhaustive coarse grid search.
    Tests ~5000 parameter combinations.
    """
    grids = {
        "soft_off":  [-3.0, -2.0, -1.5, -1.0, -0.5],
        "hard_off":  [0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
        "soft_deg":  [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
        "med_deg":   [0.02, 0.03, 0.05, 0.06, 0.08],
        "hard_deg":  [0.01, 0.02, 0.03, 0.04],
        "temp_ref":  [25.0, 30.0, 35.0],
        "temp_fac":  [0.0, 0.01, 0.02, 0.03],
    }
    
    best_loss = float("inf")
    best_p = None
    n_tested = 0
    
    # Use small sample for speed
    sample = [r for r in races[:200] if "finishing_positions" in r]
    
    total_combos = math.prod(len(v) for v in grids.values())
    print(f"Testing {total_combos} combinations on {len(sample)} races...")
    
    for soft_off in grids["soft_off"]:
        for hard_off in grids["hard_off"]:
            for soft_deg in grids["soft_deg"]:
                for med_deg in grids["med_deg"]:
                    for hard_deg in grids["hard_deg"]:
                        for temp_ref in grids["temp_ref"]:
                            for temp_fac in grids["temp_fac"]:
                                # Constraint: soft_deg > med_deg > hard_deg
                                if not (soft_deg > med_deg > hard_deg):
                                    continue
                                
                                p = [soft_off, hard_off, soft_deg, med_deg, hard_deg, temp_ref, temp_fac]
                                loss = pairwise_loss(sample, p)
                                n_tested += 1
                                
                                if loss < best_loss:
                                    best_loss = loss
                                    best_p = p[:]
                                    acc = accuracy(sample, p)
                                    if verbose:
                                        print(f"  [{n_tested}] loss={loss:.1f} acc={acc:.3f} "
                                              f"p={[round(x,4) for x in p]}")
    
    return best_p, best_loss

def fine_grid_search(races, center, radius, steps=5, verbose=True):
    """
    Fine search around a center point.
    """
    names = ["soft_off", "hard_off", "soft_deg", "med_deg", "hard_deg", "temp_ref", "temp_fac"]
    
    # Create fine grid around center
    grids = []
    for i, (name, val) in enumerate(zip(names, center)):
        r = radius[i]
        step = 2 * r / steps
        grid = [val + r * (j / steps - 0.5) for j in range(steps + 1)]
        grids.append(grid)
    
    sample = [r for r in races if "finishing_positions" in r]
    best_loss = float("inf")
    best_p = center[:]
    
    def search_recursive(idx, current):
        nonlocal best_loss, best_p
        if idx == len(grids):
            p = current[:]
            # Constraint: degs in order
            if current[2] > current[3] > current[4]:
                loss = pairwise_loss(sample, p, max_races=500)
                if loss < best_loss:
                    best_loss = loss
                    best_p = p[:]
                    acc = accuracy(sample[:200], p)
                    if verbose:
                        print(f"  loss={loss:.1f} acc={acc:.3f} p={[round(x,4) for x in p]}")
            return
        
        for val in grids[idx]:
            current.append(val)
            search_recursive(idx + 1, current)
            current.pop()
    
    search_recursive(0, [])
    return best_p, best_loss

def try_scipy_optimize(races, x0):
    """Try scipy differential evolution for global optimum."""
    try:
        from scipy.optimize import differential_evolution
        
        sample = [r for r in races[:500] if "finishing_positions" in r]
        
        bounds = [
            (-4.0, -0.1),   # soft_off
            (0.1, 3.0),     # hard_off
            (0.02, 0.40),   # soft_deg
            (0.01, 0.20),   # med_deg
            (0.005, 0.10),  # hard_deg
            (15.0, 45.0),   # temp_ref
            (-0.05, 0.05),  # temp_fac
        ]
        
        call_count = [0]
        def obj(p):
            call_count[0] += 1
            if p[2] <= p[3] or p[3] <= p[4]:
                return 1e9
            return pairwise_loss(sample, p, max_races=300)
        
        print("Running scipy differential_evolution...")
        result = differential_evolution(obj, bounds, maxiter=500, popsize=15, seed=42, disp=True)
        print(f"scipy result: loss={result.fun}, params={result.x}")
        return list(result.x), result.fun
    
    except ImportError:
        print("scipy not available, skipping")
        return x0, float("inf")

def check_if_lap_times_available(races):
    """Check whether raw lap time data is in the records."""
    for r in races[:5]:
        if "lap_times" in r or "driver_lap_times" in r or "lap_data" in r:
            return True
    return False

def extract_params_from_lap_times(races):
    """
    If raw lap times are available, fit the model directly via linear regression
    on per-lap data. This gives exact parameters.
    """
    from collections import defaultdict
    
    # Gather (tire_age, compound, adjusted_time, track_temp) for each lap
    observations = []
    
    for race in races[:100]:
        config = race["race_config"]
        base = config["base_lap_time"]
        track_temp = config["track_temp"]
        
        # Try different possible key names for lap times
        lap_times_key = None
        for key in ["lap_times", "driver_lap_times", "lap_data"]:
            if key in race:
                lap_times_key = key
                break
        
        if not lap_times_key:
            continue
        
        lap_times_data = race[lap_times_key]
        
        for _, strat in race["strategies"].items():
            driver_id = strat["driver_id"]
            
            if driver_id not in lap_times_data:
                continue
            
            laps = lap_times_data[driver_id]
            pit_sched = {p["lap"]: p["to_tire"] for p in strat.get("pit_stops", [])}
            
            tire = strat["starting_tire"]
            age = 0
            
            for lap_num, lap_time in enumerate(laps, start=1):
                age += 1   # increments BEFORE lap (regulations)
                adjusted = lap_time - base  # Remove base lap time
                observations.append({
                    "tire": tire,
                    "age": age,
                    "adjusted": adjusted,
                    "track_temp": track_temp,
                })
                if lap_num in pit_sched:
                    tire = pit_sched[lap_num]
                    age = 0
    
    if not observations:
        return None
    
    print(f"\nDirect lap time analysis: {len(observations)} observations")
    
    # For each compound, fit: adjusted = offset + deg_rate * age + temp_effect
    # Split by compound
    from collections import defaultdict
    by_compound = defaultdict(list)
    for obs in observations:
        by_compound[obs["tire"]].append(obs)
    
    params = {}
    for compound, obs in by_compound.items():
        n = len(obs)
        # Simple linear regression: adjusted = a + b*age
        # (ignoring temp for now, analyze separately)
        ages = [o["age"] for o in obs]
        vals = [o["adjusted"] for o in obs]
        
        mean_age = sum(ages) / n
        mean_val = sum(vals) / n
        
        num = sum((a - mean_age) * (v - mean_val) for a, v in zip(ages, vals))
        den = sum((a - mean_age) ** 2 for a in ages)
        
        b = num / den if den != 0 else 0
        a = mean_val - b * mean_age
        
        params[compound] = {"offset": a, "deg_rate": b, "n_obs": n}
        print(f"  {compound}: offset={a:.4f}, deg_rate={b:.6f}, n={n}")
    
    return params

def update_simulator(best_params):
    """Update the simulator.py file with calibrated parameters."""
    p = best_params
    
    # Read current simulator
    with open(SIMULATOR_PATH) as f:
        content = f.read()
    
    # Replace parameter section
    new_params = f"""# Base lap time offsets per compound (seconds relative to base_lap_time)
# SOFT = fastest (negative offset), HARD = slowest (positive offset)
# MEDIUM is the reference point (0.0)
COMPOUND_OFFSET = {{
    "SOFT":   {p.get('soft_offset', p.get('soft_off', -1.5)):.4f},   # ~{abs(p.get('soft_offset', p.get('soft_off', -1.5))):.1f}s faster per lap than medium on fresh tires
    "MEDIUM":  0.0,   # reference
    "HARD":    {p.get('hard_offset', p.get('hard_off', 0.8)):.4f},   # ~{p.get('hard_offset', p.get('hard_off', 0.8)):.1f}s slower per lap than medium on fresh tires
}}

# Tire degradation rates (seconds added per lap of tire age)
# Higher = faster degradation
DEG_RATE = {{
    "SOFT":   {p.get('soft_deg', 0.10):.4f},   # degrades fastest
    "MEDIUM": {p.get('medium_deg', p.get('med_deg', 0.05)):.4f},   # moderate degradation
    "HARD":   {p.get('hard_deg', 0.02):.4f},   # degrades slowest
}}

# Temperature model
TEMP_REF = {p.get('temp_ref', 30.0):.1f}      # Reference track temperature (degrees C)
TEMP_FACTOR = {p.get('temp_factor', p.get('temp_fac', 0.02)):.4f}   # Multiplier per degree above/below reference"""
    
    # Find and replace parameter block
    start_marker = "# Base lap time offsets per compound"
    end_marker = "TEMP_FACTOR = "
    
    start_idx = content.find(start_marker)
    end_line_start = content.find(end_marker)
    end_line_end = content.find("\n", end_line_start) + 1
    
    if start_idx >= 0 and end_line_start >= 0:
        new_content = content[:start_idx] + new_params + "\n" + content[end_line_end:]
        with open(SIMULATOR_PATH, "w") as f:
            f.write(new_content)
        print(f"\nUpdated {SIMULATOR_PATH} with calibrated parameters")
    else:
        print("WARNING: Could not find parameter section in simulator.py")
        print("Manually update these values:")
        print(f"  COMPOUND_OFFSET['SOFT'] = {p.get('soft_offset', p.get('soft_off', -1.5)):.4f}")
        print(f"  COMPOUND_OFFSET['HARD'] = {p.get('hard_offset', p.get('hard_off', 0.8)):.4f}")
        print(f"  DEG_RATE['SOFT'] = {p.get('soft_deg', 0.10):.4f}")
        print(f"  DEG_RATE['MEDIUM'] = {p.get('medium_deg', p.get('med_deg', 0.05)):.4f}")
        print(f"  DEG_RATE['HARD'] = {p.get('hard_deg', 0.02):.4f}")
        print(f"  TEMP_REF = {p.get('temp_ref', 30.0):.1f}")
        print(f"  TEMP_FACTOR = {p.get('temp_factor', p.get('temp_fac', 0.02)):.4f}")

def main():
    print("=" * 60)
    print("F1 Simulator Parameter Calibration")
    print("=" * 60)
    
    # Load data
    races = load_races(max_files=10)
    print(f"\nLoaded {len(races)} races")
    
    if not races:
        print("ERROR: No race data found in", DATA_DIR)
        sys.exit(1)
    
    # Check if lap times are directly available
    if check_if_lap_times_available(races):
        print("\n✓ Raw lap times found - using direct regression")
        direct_params = extract_params_from_lap_times(races)
        if direct_params:
            # Convert to our param format
            best_params = {
                "soft_offset": direct_params.get("SOFT", {}).get("offset", -1.5),
                "medium_offset": direct_params.get("MEDIUM", {}).get("offset", 0.0),
                "hard_offset": direct_params.get("HARD", {}).get("offset", 0.8),
                "soft_deg": direct_params.get("SOFT", {}).get("deg_rate", 0.10),
                "medium_deg": direct_params.get("MEDIUM", {}).get("deg_rate", 0.05),
                "hard_deg": direct_params.get("HARD", {}).get("deg_rate", 0.02),
                "temp_ref": 30.0,
                "temp_factor": 0.02,
            }
            # Then optimize temp params with these fixed
            # ... (further fine-tuning)
    else:
        print("\n✓ No raw lap times - using ranking-based optimization")
        
        # Phase 1: Coarse grid search
        print("\n--- Phase 1: Coarse Grid Search ---")
        best_p, best_loss = coarse_grid_search(races)
        
        acc = accuracy(races[:500], best_p)
        print(f"\nCoarse best: loss={best_loss:.1f} acc={acc:.3f}")
        print(f"Params: {[round(x, 4) for x in best_p]}")
        
        # Phase 2: scipy optimization
        print("\n--- Phase 2: Global Optimization ---")
        scipy_p, scipy_loss = try_scipy_optimize(races, best_p)
        
        if scipy_loss < best_loss:
            best_p = scipy_p
            best_loss = scipy_loss
            print("scipy found better params!")
        
        best_params = {
            "soft_offset": best_p[0],
            "medium_offset": 0.0,
            "hard_offset": best_p[1],
            "soft_deg": best_p[2],
            "medium_deg": best_p[3],
            "hard_deg": best_p[4],
            "temp_ref": best_p[5],
            "temp_factor": best_p[6],
        }
    
    # Evaluate on full dataset
    all_races = load_races(max_files=30)
    p_list = [
        best_params["soft_offset"],
        best_params["hard_offset"],
        best_params["soft_deg"],
        best_params["medium_deg"],
        best_params["hard_deg"],
        best_params["temp_ref"],
        best_params["temp_factor"],
    ]
    final_acc = accuracy(all_races, p_list)
    print(f"\n=== Final Accuracy on Full Dataset: {final_acc:.1%} ===")
    
    # Save params
    os.makedirs("analysis", exist_ok=True)
    with open(PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved params to {PARAMS_PATH}")
    
    # Update simulator
    update_simulator(best_params)
    
    print("\nCalibration complete!")
    print(f"Run ./test_runner.sh to evaluate on test cases.")

if __name__ == "__main__":
    main()
