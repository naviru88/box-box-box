#!/usr/bin/env python3
"""
Analyze historical race data to reverse-engineer lap time formulas.

Run this script against the historical data to find:
1. Tire compound base offsets (SOFT/MEDIUM/HARD vs base lap time)
2. Tire degradation model (how lap time increases with tire age)
3. Temperature effects on tire behavior
4. Pit stop mechanics verification

Usage:
    python3 analysis/analyze.py
"""

import json
import glob
import os
import math
from collections import defaultdict

DATA_DIR = "data/historical_races"

def load_races(max_files=5):
    """Load a sample of historical races for analysis."""
    races = []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    for f in files[:max_files]:
        with open(f) as fh:
            batch = json.load(fh)
            if isinstance(batch, list):
                races.extend(batch)
            else:
                races.append(batch)
    print(f"Loaded {len(races)} races from {min(max_files, len(files))} files")
    return races

def simulate_race_time(race, params):
    """
    Simulate total race time for all drivers given tire model params.
    
    params dict keys:
        soft_offset: lap time delta for SOFT vs base (negative = faster)
        medium_offset: lap time delta for MEDIUM vs base
        hard_offset: lap time delta for HARD vs base
        soft_deg: seconds added per lap of tire age for SOFT
        medium_deg: seconds added per lap of tire age for MEDIUM
        hard_deg: seconds added per lap of tire age for HARD
        temp_factor: multiplier on degradation based on track temp
        temp_ref: reference temperature for normalization
    """
    config = race["race_config"]
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    track_temp = config["track_temp"]
    
    soft_offset = params.get("soft_offset", -1.0)
    medium_offset = params.get("medium_offset", 0.0)
    hard_offset = params.get("hard_offset", 0.6)
    soft_deg = params.get("soft_deg", 0.08)
    medium_deg = params.get("medium_deg", 0.04)
    hard_deg = params.get("hard_deg", 0.02)
    temp_ref = params.get("temp_ref", 30.0)
    temp_factor = params.get("temp_factor", 0.02)
    
    # Temperature multiplier on degradation
    temp_mult = 1.0 + (track_temp - temp_ref) * temp_factor
    
    compound_offset = {"SOFT": soft_offset, "MEDIUM": medium_offset, "HARD": hard_offset}
    compound_deg = {"SOFT": soft_deg, "MEDIUM": medium_deg, "HARD": hard_deg}
    
    driver_times = {}
    
    for pos_key, strategy in race["strategies"].items():
        driver_id = strategy["driver_id"]
        current_tire = strategy["starting_tire"]
        tire_age = 0
        total_time = 0.0
        
        # Build pit stop lookup: lap -> new tire
        pit_laps = {}
        for pit in strategy.get("pit_stops", []):
            pit_laps[pit["lap"]] = pit["to_tire"]
        
        for lap in range(1, total_laps + 1):
            # Check if pitting THIS lap (pit stop happens at END of lap)
            # Pit stop penalty added when pitting
            
            # Age increments BEFORE lap time calc (regulations.md)
            tire_age += 1

            # Calculate lap time for this lap
            offset = compound_offset[current_tire]
            deg_rate = compound_deg[current_tire]
            lap_time = base + offset + (tire_age * deg_rate * temp_mult)
            total_time += lap_time

            # Pit stop: occurs after completing this lap
            if lap in pit_laps:
                total_time += pit_time
                current_tire = pit_laps[lap]
                tire_age = 0   # reset; next lap will start at age=1
        
        driver_times[driver_id] = total_time
    
    # Sort by total time
    finishing_order = sorted(driver_times.keys(), key=lambda d: driver_times[d])
    return finishing_order, driver_times

def evaluate_params(races, params, verbose=False):
    """Evaluate how well params predict historical finishing orders."""
    correct = 0
    total = 0
    
    for race in races:
        if "finishing_positions" not in race:
            continue
        
        predicted_order, _ = simulate_race_time(race, params)
        actual_order = race["finishing_positions"]
        
        if predicted_order == actual_order:
            correct += 1
        elif verbose and total < 3:
            print(f"\nRace {race.get('race_id', '?')}:")
            print(f"  Predicted: {predicted_order[:5]}")
            print(f"  Actual:    {actual_order[:5]}")
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total

def analyze_single_race(race, params):
    """Deep-dive analysis of one race to check model fit."""
    config = race["race_config"]
    print(f"\nRace: {race.get('race_id', '?')}")
    print(f"Track: {config['track']}, Laps: {config['total_laps']}, "
          f"Base: {config['base_lap_time']}s, Pit: {config['pit_lane_time']}s, "
          f"Temp: {config['track_temp']}°C")
    
    _, driver_times = simulate_race_time(race, params)
    actual = race.get("finishing_positions", [])
    
    print("\nDriver times (top 5):")
    sorted_drivers = sorted(driver_times.items(), key=lambda x: x[1])
    for i, (d, t) in enumerate(sorted_drivers[:5]):
        marker = "✓" if i < len(actual) and actual[i] == d else "✗"
        print(f"  {marker} P{i+1}: {d} = {t:.3f}s")
    
    print(f"\nActual top 5: {actual[:5]}")

def grid_search(races, param_grid):
    """Simple grid search over parameter space."""
    best_acc = -1
    best_params = None
    
    results = []
    
    for soft_off in param_grid["soft_offset"]:
        for med_off in param_grid["medium_offset"]:
            for hard_off in param_grid["hard_offset"]:
                for s_deg in param_grid["soft_deg"]:
                    for m_deg in param_grid["medium_deg"]:
                        for h_deg in param_grid["hard_deg"]:
                            for tf in param_grid["temp_factor"]:
                                params = {
                                    "soft_offset": soft_off,
                                    "medium_offset": med_off,
                                    "hard_offset": hard_off,
                                    "soft_deg": s_deg,
                                    "medium_deg": m_deg,
                                    "hard_deg": h_deg,
                                    "temp_ref": 30.0,
                                    "temp_factor": tf,
                                }
                                acc, correct, total = evaluate_params(races, params)
                                results.append((acc, params.copy()))
                                if acc > best_acc:
                                    best_acc = acc
                                    best_params = params.copy()
                                    print(f"New best: {acc:.3f} ({correct}/{total}) | "
                                          f"s={soft_off}, m={med_off}, h={hard_off}, "
                                          f"sd={s_deg}, md={m_deg}, hd={h_deg}, tf={tf}")
    
    return best_params, best_acc, results

def analyze_tire_degradation(races):
    """
    Try to directly infer tire degradation from race data.
    
    For a driver with no pit stops, their lap time increase over laps 
    tells us the degradation rate directly.
    
    For races where we can reconstruct lap times from finishing positions
    and race time differences, we can calibrate.
    """
    print("\n=== Tire Degradation Analysis ===")
    
    # Look for races with drivers who don't pit (1-stop strategy gives us stint data)
    compound_samples = defaultdict(list)
    
    for race in races[:100]:
        config = race["race_config"]
        base = config["base_lap_time"]
        total_laps = config["total_laps"]
        
        for pos_key, strat in race["strategies"].items():
            pits = strat.get("pit_stops", [])
            
            # Single stint (no pits) - we know the entire tire usage
            if len(pits) == 0:
                tire = strat["starting_tire"]
                # This driver ran tire for total_laps laps
                compound_samples[tire].append({
                    "laps": total_laps,
                    "tire": tire,
                    "race_id": race.get("race_id"),
                })
    
    for compound, samples in compound_samples.items():
        print(f"\n{compound}: {len(samples)} no-stop stints found")
        if samples:
            avg_laps = sum(s["laps"] for s in samples) / len(samples)
            print(f"  Avg stint length: {avg_laps:.1f} laps")

def check_race_structure(races):
    """Examine the structure of race records."""
    if not races:
        return
    
    r = races[0]
    print("\n=== Race Structure ===")
    print(json.dumps({k: v for k, v in r.items() if k != "strategies"}, indent=2))
    
    print("\n=== Sample Strategy (pos1) ===")
    print(json.dumps(r["strategies"]["pos1"], indent=2))
    
    print(f"\n=== Keys in race record ===")
    print(list(r.keys()))
    
    if "finishing_positions" in r:
        print(f"\nFinishing positions present: {r['finishing_positions'][:5]}...")
    if "lap_times" in r:
        print(f"\nLap times present for driver: {list(r['lap_times'].keys())[0]}")
        first_driver = list(r['lap_times'].keys())[0]
        print(f"  First 5 laps: {r['lap_times'][first_driver][:5]}")

def main():
    print("=== F1 Race Simulator - Data Analysis ===\n")
    
    # Load sample of races
    races = load_races(max_files=2)
    
    if not races:
        print("ERROR: No race data found. Check DATA_DIR path.")
        return
    
    # 1. Check structure
    check_race_structure(races)
    
    # 2. Analyze tire stints
    analyze_tire_degradation(races)
    
    # 3. Quick evaluation with default params
    default_params = {
        "soft_offset": -1.0,
        "medium_offset": 0.0,
        "hard_offset": 0.5,
        "soft_deg": 0.08,
        "medium_deg": 0.04,
        "hard_deg": 0.02,
        "temp_ref": 30.0,
        "temp_factor": 0.02,
    }
    
    print("\n=== Default Parameter Evaluation ===")
    acc, correct, total = evaluate_params(races, default_params, verbose=True)
    print(f"\nAccuracy: {acc:.1%} ({correct}/{total})")
    
    # 4. Analyze first race in detail
    if races:
        analyze_single_race(races[0], default_params)
    
    # 5. Grid search (coarse)
    print("\n=== Grid Search ===")
    param_grid = {
        "soft_offset":  [-2.0, -1.5, -1.0, -0.5],
        "medium_offset": [0.0],
        "hard_offset":   [0.3, 0.5, 0.8, 1.0, 1.5],
        "soft_deg":      [0.05, 0.08, 0.10, 0.12, 0.15],
        "medium_deg":    [0.03, 0.04, 0.05, 0.06],
        "hard_deg":      [0.01, 0.02, 0.03],
        "temp_factor":   [0.0, 0.01, 0.02, 0.03],
    }
    
    best_params, best_acc, _ = grid_search(races, param_grid)
    print(f"\nBest params: {best_params}")
    print(f"Best accuracy: {best_acc:.1%}")
    
    # Save best params
    with open("analysis/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print("\nSaved best_params.json")

if __name__ == "__main__":
    main()
