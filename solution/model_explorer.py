#!/usr/bin/env python3
"""
Model Explorer: Try different functional forms for tire degradation.

The basic model assumes linear degradation. But maybe it's:
- Quadratic: deg = a*age + b*age^2
- Exponential: deg = a * (e^(b*age) - 1)  
- Cliff: sudden performance drop after a certain age
- Step function: deg changes at certain lap counts

Also explores:
- Whether pit stop includes in-lap + out-lap time (not just pit_lane_time)
- Whether temperature affects offset (not just degradation)
- Whether compounds have different behaviors on different tracks

Run AFTER seeing some accuracy numbers to understand where errors come from.
"""

import json
import glob
import os
import sys
import math
from collections import defaultdict

DATA_DIR = "data/historical_races"

def load_races(max_files=5):
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

# ---- Different model variants ----

def linear_deg(tire_age, a, b):
    """Linear: offset + a*age"""
    return a * tire_age

def quadratic_deg(tire_age, a, b):
    """Quadratic: a*age + b*age^2"""
    return a * tire_age + b * tire_age * tire_age

def exponential_deg(tire_age, a, b):
    """Exponential: a * (exp(b*age) - 1)"""
    return a * (math.exp(b * tire_age) - 1)

def sim_with_model(strategy, config, params, deg_model="linear"):
    """Simulate with different degradation model."""
    base = config["base_lap_time"]
    total_laps = config["total_laps"]
    pit_time = config["pit_lane_time"]
    track_temp = config["track_temp"]
    
    # params layout for linear: [soft_off, hard_off, soft_a, soft_b, med_a, med_b, hard_a, hard_b, temp_ref, temp_fac]
    # For linear: b terms are ignored
    # For quadratic: both a and b matter
    
    soft_off = params[0]
    hard_off = params[1]
    
    deg_params = {
        "SOFT":   (params[2], params[3]),
        "MEDIUM": (params[4], params[5]),
        "HARD":   (params[6], params[7]),
    }
    
    temp_ref = params[8]
    temp_fac = params[9]
    temp_mult = 1.0 + (track_temp - temp_ref) * temp_fac
    
    offsets = {"SOFT": soft_off, "MEDIUM": 0.0, "HARD": hard_off}
    
    deg_fn = {
        "linear": linear_deg,
        "quadratic": quadratic_deg,
        "exponential": exponential_deg,
    }[deg_model]
    
    pit_sched = {p["lap"]: p["to_tire"] for p in strategy.get("pit_stops", [])}
    
    tire = strategy["starting_tire"]
    age = 0
    total = 0.0
    
    for lap in range(1, total_laps + 1):
        age += 1   # increments BEFORE lap time (regulations)
        a, b = deg_params[tire]
        deg = deg_fn(age, a, b) * temp_mult
        total += base + offsets[tire] + deg
        
        if lap in pit_sched:
            total += pit_time
            tire = pit_sched[lap]
            age = 0
    
    return total

def predict_order_model(race, params, deg_model):
    config = race["race_config"]
    times = {}
    for _, strat in race["strategies"].items():
        did = strat["driver_id"]
        times[did] = sim_with_model(strat, config, params, deg_model)
    return sorted(times.keys(), key=lambda d: times[d])

def accuracy_model(races, params, deg_model, max_races=500):
    correct = 0
    total = 0
    for race in races[:max_races]:
        if "finishing_positions" not in race:
            continue
        predicted = predict_order_model(race, params, deg_model)
        if predicted == race["finishing_positions"]:
            correct += 1
        total += 1
    return correct / total if total else 0

def check_pit_stop_model(races):
    """
    Does the pit_lane_time fully account for the pit stop cost?
    Or is there an additional in-lap + out-lap time?
    
    Test: find cases where a driver pits very early (lap 1-5) on SOFT to HARD.
    They should be at a big disadvantage if pit costs more than pit_lane_time.
    """
    print("\n=== Pit Stop Model Analysis ===")
    
    early_pit_results = []
    
    for race in races[:200]:
        if "finishing_positions" not in race:
            continue
        
        config = race["race_config"]
        actual = race["finishing_positions"]
        
        for _, strat in race["strategies"].items():
            pits = strat.get("pit_stops", [])
            if pits and pits[0]["lap"] <= 5:
                driver = strat["driver_id"]
                pos = actual.index(driver) + 1 if driver in actual else None
                early_pit_results.append({
                    "race_id": race.get("race_id"),
                    "driver": driver,
                    "pit_lap": pits[0]["lap"],
                    "from": pits[0]["from_tire"],
                    "to": pits[0]["to_tire"],
                    "pos": pos,
                    "n_stops": len(pits),
                })
    
    print(f"Found {len(early_pit_results)} early pit stops (lap 1-5)")
    if early_pit_results:
        avg_pos = sum(r["pos"] for r in early_pit_results if r["pos"]) / len([r for r in early_pit_results if r["pos"]])
        print(f"Average finishing position: {avg_pos:.1f} (expected: ~10.5 if random)")
        print("Sample:")
        for r in early_pit_results[:10]:
            print(f"  Race {r['race_id']}: lap {r['pit_lap']} {r['from']}->{r['to']}, "
                  f"P{r['pos']}, total {r['n_stops']} stops")

def check_temperature_effects(races):
    """
    Analyze whether temperature affects compound offset (not just degradation).
    
    If at very high temp, all strategies perform worse on SOFT equally,
    we'd expect the same relative ordering. But if high temp makes SOFT
    specifically much worse, we'd see ranking changes.
    """
    print("\n=== Temperature Effect Analysis ===")
    
    by_temp = defaultdict(lambda: defaultdict(list))
    
    for race in races[:500]:
        config = race["race_config"]
        temp = config["track_temp"]
        temp_bucket = (temp // 5) * 5  # bucket by 5°C
        
        # For each driver, record: starting compound and finishing position
        actual = race.get("finishing_positions", [])
        for _, strat in race["strategies"].items():
            driver = strat["driver_id"]
            compound = strat["starting_tire"]
            if driver in actual:
                pos = actual.index(driver) + 1
                by_temp[temp_bucket][compound].append(pos)
    
    print(f"\nAverage finishing position by temp bucket and starting compound:")
    print(f"{'Temp':>6} | {'SOFT':>6} {'MEDIUM':>8} {'HARD':>6} | {'n_races':>8}")
    print("-" * 45)
    
    for temp in sorted(by_temp.keys()):
        compounds = by_temp[temp]
        n_soft = len(compounds.get("SOFT", []))
        n_med = len(compounds.get("MEDIUM", []))
        n_hard = len(compounds.get("HARD", []))
        
        avg_soft = sum(compounds.get("SOFT", [0])) / max(n_soft, 1)
        avg_med = sum(compounds.get("MEDIUM", [0])) / max(n_med, 1)
        avg_hard = sum(compounds.get("HARD", [0])) / max(n_hard, 1)
        
        n_races = max(n_soft, n_med, n_hard) // 3
        
        print(f"{temp:>5}°C | {avg_soft:>6.1f} {avg_med:>8.1f} {avg_hard:>6.1f} | {n_races:>8}")

def check_track_specific_params(races):
    """
    Do different tracks have different tire behavior?
    (e.g., high-speed tracks like Monza vs technical tracks like Monaco)
    """
    print("\n=== Track-Specific Analysis ===")
    
    by_track = defaultdict(lambda: {"races": 0, "strategies": defaultdict(int)})
    
    for race in races[:200]:
        track = race["race_config"]["track"]
        by_track[track]["races"] += 1
        
        for _, strat in race["strategies"].items():
            pits = strat.get("pit_stops", [])
            n_stops = len(pits)
            compounds = strat["starting_tire"]
            by_track[track]["strategies"][n_stops] += 1
    
    print(f"\nTracks in dataset:")
    for track, data in sorted(by_track.items())[:20]:
        stops_dist = data["strategies"]
        total_drivers = sum(stops_dist.values())
        avg_stops = sum(k*v for k, v in stops_dist.items()) / max(total_drivers, 1)
        print(f"  {track}: {data['races']} races, avg {avg_stops:.1f} stops/driver")

def check_model_residuals(races, params_dict):
    """
    After running the linear model, look at residuals to understand
    what systematic errors remain.
    """
    print("\n=== Model Residual Analysis ===")
    
    from simulator import compute_driver_total_time
    
    wrong_pairs = defaultdict(int)  # (pred_compound, actual_compound) -> count
    
    for race in races[:200]:
        if "finishing_positions" not in race:
            continue
        
        config = race["race_config"]
        actual = race["finishing_positions"]
        
        times = {}
        strats_by_driver = {}
        for _, strat in race["strategies"].items():
            did = strat["driver_id"]
            times[did] = compute_driver_total_time(strat, config)
            strats_by_driver[did] = strat
        
        predicted = sorted(times.keys(), key=lambda d: times[d])
        
        if predicted == actual:
            continue
        
        # Find first wrong pair
        for i in range(len(actual)):
            if i < len(predicted) and predicted[i] != actual[i]:
                pred_driver = predicted[i]
                actual_driver = actual[i]
                
                pred_strat = strats_by_driver.get(pred_driver, {})
                actual_strat = strats_by_driver.get(actual_driver, {})
                
                pred_tire = pred_strat.get("starting_tire", "?")
                actual_tire = actual_strat.get("starting_tire", "?")
                
                key = f"pred={pred_tire}_actual={actual_tire}"
                wrong_pairs[key] += 1
                break
    
    print("\nMost common wrong predictions at P1:")
    for k, v in sorted(wrong_pairs.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v} times")

def main():
    print("=" * 60)
    print("Model Explorer")
    print("=" * 60)
    
    races = load_races(max_files=3)
    print(f"Loaded {len(races)} races")
    
    if not races:
        print("No races found!")
        return
    
    # Check race structure for additional fields
    r = races[0]
    print(f"\nRace keys: {list(r.keys())}")
    print(f"Config keys: {list(r['race_config'].keys())}")
    
    # Run all analyses
    check_pit_stop_model(races)
    check_temperature_effects(races)
    check_track_specific_params(races)
    
    # Test different model variants if scipy available
    try:
        from scipy.optimize import minimize
        print("\n=== Testing Model Variants ===")
        
        sample = [r for r in races[:100] if "finishing_positions" in r]
        
        # Test linear vs quadratic
        # Linear baseline params: [soft_off, hard_off, soft_a, 0, med_a, 0, hard_a, 0, temp_ref, temp_fac]
        linear_p = [-1.5, 0.8, 0.10, 0.0, 0.05, 0.0, 0.02, 0.0, 30.0, 0.01]
        
        acc_linear = accuracy_model(sample, linear_p, "linear")
        print(f"Linear model accuracy: {acc_linear:.3f}")
        
        quadratic_p = [-1.5, 0.8, 0.08, 0.001, 0.04, 0.0005, 0.015, 0.0002, 30.0, 0.01]
        acc_quad = accuracy_model(sample, quadratic_p, "quadratic")
        print(f"Quadratic model accuracy: {acc_quad:.3f}")
        
        print("\nNote: If quadratic >> linear, consider using quadratic degradation")
    
    except ImportError:
        pass
    
    print("\nDone!")

if __name__ == "__main__":
    main()
