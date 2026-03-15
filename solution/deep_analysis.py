#!/usr/bin/env python3
"""
Deep analysis: extract the EXACT tire model from historical data.

Key insight: Since outcomes are deterministic, we can use optimization
to find the exact parameters. With enough races, parameter estimation
becomes very accurate.

Strategy:
1. For races where we know finishing positions, the ranking of total times
   must match exactly.
2. We can set up inequalities: winner_time < 2nd_time < ... < 20th_time
3. Use these constraints to narrow down parameters.

Better approach: Look for "natural experiments" in the data:
- Two drivers with identical strategies except one variable
- Races with only one compound used (pure SOFT race, etc.)
- Compare 1-stop vs 2-stop with same compounds
"""

import json
import glob
import os
import math
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize, differential_evolution

DATA_DIR = "data/historical_races"

def load_all_races(max_files=30):
    """Load races for parameter fitting."""
    races = []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    for f in files[:max_files]:
        with open(f) as fh:
            batch = json.load(fh)
            if isinstance(batch, list):
                races.extend(batch)
            else:
                races.append(batch)
    return races

def compute_driver_total_time(strategy, config, params):
    """Compute total race time for a single driver."""
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    track_temp = config["track_temp"]
    
    soft_offset = params[0]
    medium_offset = 0.0  # Reference point
    hard_offset = params[1]
    soft_deg = params[2]
    medium_deg = params[3]
    hard_deg = params[4]
    temp_ref = params[5]
    temp_factor = params[6]
    
    temp_mult = 1.0 + (track_temp - temp_ref) * temp_factor
    
    compound_offset = {"SOFT": soft_offset, "MEDIUM": medium_offset, "HARD": hard_offset}
    compound_deg = {"SOFT": soft_deg, "MEDIUM": medium_deg, "HARD": hard_deg}
    
    current_tire = strategy["starting_tire"]
    tire_age = 0
    total_time = 0.0
    
    pit_laps = {}
    for pit in strategy.get("pit_stops", []):
        pit_laps[pit["lap"]] = pit["to_tire"]
    
    for lap in range(1, total_laps + 1):
        offset = compound_offset[current_tire]
        deg_rate = compound_deg[current_tire]
        tire_age += 1   # increments BEFORE lap time (regulations)
        lap_time = base + offset + (tire_age * deg_rate * temp_mult)
        total_time += lap_time
        
        if lap in pit_laps:
            total_time += pit_time
            current_tire = pit_laps[lap]
            tire_age = 0
    
    return total_time

def ranking_loss(params, races, sample_size=500):
    """
    Loss function: number of incorrectly ordered pairs.
    Minimizing this maximizes exact-order accuracy.
    """
    loss = 0.0
    sample = races[:sample_size]
    
    for race in sample:
        if "finishing_positions" not in race:
            continue
        
        config = race["race_config"]
        actual_order = race["finishing_positions"]
        
        # Compute all driver times
        times = {}
        for pos_key, strat in race["strategies"].items():
            driver_id = strat["driver_id"]
            times[driver_id] = compute_driver_total_time(strat, config, params)
        
        # Check if ranking matches
        predicted = sorted(times.keys(), key=lambda d: times[d])
        
        if predicted != actual_order:
            # Count pairwise violations (soft loss)
            for i in range(len(actual_order)):
                for j in range(i + 1, len(actual_order)):
                    di = actual_order[i]
                    dj = actual_order[j]
                    if di in times and dj in times:
                        # di should be faster (less time) than dj
                        if times[di] >= times[dj]:
                            loss += 1.0
    
    return loss

def exact_accuracy(params, races):
    """Count exact order matches."""
    correct = 0
    total = 0
    for race in races:
        if "finishing_positions" not in race:
            continue
        config = race["race_config"]
        times = {}
        for pos_key, strat in race["strategies"].items():
            driver_id = strat["driver_id"]
            times[driver_id] = compute_driver_total_time(strat, config, params)
        predicted = sorted(times.keys(), key=lambda d: times[d])
        if predicted == race["finishing_positions"]:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

def analyze_pit_stop_timing(races):
    """
    Key insight: if two drivers have the same strategy but pit on different laps,
    the one who pits later should be slower (more tire deg) but saves a pit stop
    time difference... Actually they have the SAME pit stop penalty.
    
    Better: look at cases where two drivers have IDENTICAL strategies
    and verify they tie (impossible with 20 unique drivers... unless strategies repeat)
    """
    print("\n=== Pit Stop Analysis ===")
    
    # Find races where we can isolate tire compound effects
    no_pit_races_by_compound = defaultdict(list)
    
    for race in races[:200]:
        config = race["race_config"]
        for pos_key, strat in race["strategies"].items():
            if len(strat.get("pit_stops", [])) == 0:
                compound = strat["starting_tire"]
                no_pit_races_by_compound[compound].append({
                    "total_laps": config["total_laps"],
                    "base_lap_time": config["base_lap_time"],
                    "track_temp": config["track_temp"],
                    "driver": strat["driver_id"],
                    "race_id": race.get("race_id"),
                    "finishing_pos": race.get("finishing_positions", []).index(strat["driver_id"]) + 1
                    if strat["driver_id"] in race.get("finishing_positions", []) else None
                })
    
    for compound, samples in no_pit_races_by_compound.items():
        print(f"\n{compound} no-stop drivers: {len(samples)} samples")
        if samples[:3]:
            for s in samples[:3]:
                print(f"  Race {s['race_id']}: {s['total_laps']} laps, "
                      f"base={s['base_lap_time']}, temp={s['track_temp']}, "
                      f"finished P{s['finishing_pos']}")

def find_natural_experiments(races):
    """
    Find pairs of drivers in the same race with strategies that differ
    in only one variable, allowing us to isolate that variable's effect.
    """
    print("\n=== Natural Experiments ===")
    
    isolation_cases = []
    
    for race in races[:100]:
        if "finishing_positions" not in race:
            continue
        
        config = race["race_config"]
        strategies = race["strategies"]
        driver_list = list(strategies.values())
        
        for i in range(len(driver_list)):
            for j in range(i + 1, len(driver_list)):
                s1 = driver_list[i]
                s2 = driver_list[j]
                
                # Check if they have same tire compound throughout but different pit laps
                pits1 = s1.get("pit_stops", [])
                pits2 = s2.get("pit_stops", [])
                
                if len(pits1) == 1 and len(pits2) == 1:
                    # Same number of stops
                    same_start = s1["starting_tire"] == s2["starting_tire"]
                    same_compounds = (pits1[0]["to_tire"] == pits2[0]["to_tire"])
                    diff_timing = pits1[0]["lap"] != pits2[0]["lap"]
                    
                    if same_start and same_compounds and diff_timing:
                        # Only difference: when they pit
                        actual = race["finishing_positions"]
                        pos1 = actual.index(s1["driver_id"]) + 1 if s1["driver_id"] in actual else None
                        pos2 = actual.index(s2["driver_id"]) + 1 if s2["driver_id"] in actual else None
                        
                        isolation_cases.append({
                            "race_id": race.get("race_id"),
                            "d1": s1["driver_id"], "pit_lap_1": pits1[0]["lap"],
                            "d2": s2["driver_id"], "pit_lap_2": pits2[0]["lap"],
                            "start_tire": s1["starting_tire"],
                            "end_tire": pits1[0]["to_tire"],
                            "pos1": pos1, "pos2": pos2,
                            "total_laps": config["total_laps"],
                        })
    
    print(f"Found {len(isolation_cases)} same-compound, diff-timing pairs")
    
    # The driver who pits LATER runs longer on first stint
    # If first stint has more deg than second, better to pit EARLIER
    # If second stint has less deg, earlier pit = more laps on fresh tire
    
    # Count: when does pitting LATER win?
    later_wins = 0
    for case in isolation_cases[:50]:
        if case["pos1"] and case["pos2"]:
            if case["pit_lap_2"] > case["pit_lap_1"]:
                # d2 pits later
                if case["pos2"] < case["pos1"]:  # lower pos = better
                    later_wins += 1
            print(f"  Pit {case['pit_lap_1']} vs {case['pit_lap_2']} -> "
                  f"P{case['pos1']} vs P{case['pos2']} "
                  f"({case['start_tire']} -> {case['end_tire']}, {case['total_laps']} laps)")
    
    return isolation_cases

def optimize_params(races, n_races=1000):
    """Use scipy optimization to find best parameters."""
    print(f"\n=== Parameter Optimization (using {n_races} races) ===")
    
    sample = [r for r in races[:n_races] if "finishing_positions" in r]
    print(f"Using {len(sample)} races with known results")
    
    # params: [soft_offset, hard_offset, soft_deg, medium_deg, hard_deg, temp_ref, temp_factor]
    # medium_offset = 0 (reference)
    
    # Initial guess
    x0 = [-1.0, 0.5, 0.08, 0.04, 0.02, 30.0, 0.01]
    
    # Bounds
    bounds = [
        (-5.0, 0.0),    # soft_offset (must be negative = faster than medium)
        (0.0, 3.0),     # hard_offset (must be positive = slower than medium)
        (0.01, 0.5),    # soft_deg
        (0.01, 0.3),    # medium_deg
        (0.005, 0.2),   # hard_deg
        (15.0, 45.0),   # temp_ref
        (-0.1, 0.1),    # temp_factor
    ]
    
    call_count = [0]
    
    def objective(params):
        call_count[0] += 1
        loss = ranking_loss(params, sample, sample_size=min(200, len(sample)))
        if call_count[0] % 100 == 0:
            acc = exact_accuracy(params, sample[:100])
            print(f"  Iter {call_count[0]}: loss={loss:.1f}, acc={acc:.3f}, params={[f'{p:.4f}' for p in params]}")
        return loss
    
    # Coarse grid search first
    print("Running differential evolution...")
    result = differential_evolution(
        objective, bounds, 
        maxiter=200, 
        popsize=10,
        tol=0.01,
        seed=42,
        disp=True
    )
    
    best_params = result.x
    print(f"\nOptimized params: {best_params}")
    print(f"Final loss: {result.fun}")
    
    acc = exact_accuracy(best_params, sample)
    print(f"Accuracy on training set: {acc:.3f}")
    
    return best_params

def main():
    print("=== Deep Analysis: Extracting Exact Tire Model ===\n")
    
    races = load_all_races(max_files=5)
    print(f"Loaded {len(races)} races")
    
    if not races:
        print("No races found!")
        return
    
    # Check race structure for lap_times data
    r = races[0]
    print(f"Race keys: {list(r.keys())}")
    if "lap_times" in r:
        print("LAP TIMES DATA AVAILABLE! This is gold.")
        # Extract raw lap times for direct analysis
        analyze_lap_times_direct(races)
    else:
        print("No lap_times in data - will use ranking-based optimization")
    
    # Natural experiments
    find_natural_experiments(races)
    
    # Optimization
    best_params = optimize_params(races, n_races=2000)
    
    params_dict = {
        "soft_offset": best_params[0],
        "medium_offset": 0.0,
        "hard_offset": best_params[1],
        "soft_deg": best_params[2],
        "medium_deg": best_params[3],
        "hard_deg": best_params[4],
        "temp_ref": best_params[5],
        "temp_factor": best_params[6],
    }
    
    os.makedirs("analysis", exist_ok=True)
    with open("analysis/optimized_params.json", "w") as f:
        json.dump(params_dict, f, indent=2)
    print(f"\nSaved optimized params: {params_dict}")

def analyze_lap_times_direct(races):
    """If lap_times are available in the data, extract degradation directly."""
    print("\n=== Direct Lap Time Analysis ===")
    
    deg_by_compound = defaultdict(list)
    
    for race in races[:50]:
        if "lap_times" not in race:
            continue
        
        config = race["race_config"]
        base = config["base_lap_time"]
        
        for pos_key, strat in race["strategies"].items():
            driver_id = strat["driver_id"]
            if driver_id not in race["lap_times"]:
                continue
            
            laps_data = race["lap_times"][driver_id]
            
            # Reconstruct tire stints
            pit_laps = {p["lap"]: p["to_tire"] for p in strat.get("pit_stops", [])}
            current_tire = strat["starting_tire"]
            tire_age = 0
            
            for lap_idx, lap_time in enumerate(laps_data):
                lap_num = lap_idx + 1
                
                # Age increments BEFORE lap time (regulations)
                tire_age += 1

                # Record: (tire_age, lap_time - base) for this compound
                adjusted = lap_time - base
                deg_by_compound[current_tire].append({
                    "tire_age": tire_age,
                    "adjusted_time": adjusted,
                    "track_temp": config["track_temp"],
                })
                
                if lap_num in pit_laps:
                    current_tire = pit_laps[lap_num]
                    tire_age = 0
    
    for compound, data in deg_by_compound.items():
        if not data:
            continue
        # Fit linear: adjusted_time = offset + deg_rate * tire_age
        ages = [d["tire_age"] for d in data]
        times = [d["adjusted_time"] for d in data]
        
        if len(ages) > 1:
            n = len(ages)
            sum_x = sum(ages)
            sum_y = sum(times)
            sum_xy = sum(x*y for x, y in zip(ages, times))
            sum_x2 = sum(x*x for x in ages)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            intercept = (sum_y - slope * sum_x) / n
            
            print(f"\n{compound}:")
            print(f"  Offset (lap 0): {intercept:.4f}s vs base")
            print(f"  Degradation rate: {slope:.4f}s/lap")
            print(f"  Sample size: {n} laps")

if __name__ == "__main__":
    main()
