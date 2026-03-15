#!/usr/bin/env python3
"""
Read actual historical races and figure out the EXACT model by inspection.
Run: python3 solution/inspect_data.py
"""
import json, glob, sys
from collections import defaultdict

DATA_DIR = "data/historical_races"

def load_races(n_files=1):
    races = []
    for f in sorted(glob.glob(f"{DATA_DIR}/*.json"))[:n_files]:
        data = json.load(open(f))
        races.extend(data if isinstance(data, list) else [data])
    return races

def print_race(r):
    """Print a race in full detail."""
    config = r["race_config"]
    print(f"\nRace: {r['race_id']}")
    print(f"  Track={config['track']}, Laps={config['total_laps']}, "
          f"Base={config['base_lap_time']}, Pit={config['pit_lane_time']}, "
          f"Temp={config['track_temp']}")
    
    actual = r.get("finishing_positions", [])
    print(f"  Result: {actual[:5]}...")
    
    # Print each driver's strategy and finishing position
    print(f"\n  {'Pos':>4} {'Driver':>8} {'Start':>8} {'Stops':>6}  Strategy")
    for i, did in enumerate(actual[:20]):
        for _, s in r["strategies"].items():
            if s["driver_id"] == did:
                pits = s.get("pit_stops", [])
                strat = s["starting_tire"]
                for p in pits:
                    strat += f"→[L{p['lap']}]{p['to_tire']}"
                print(f"  {i+1:>4} {did:>8} {s['starting_tire']:>8} {len(pits):>6}  {strat}")
                break

def find_identical_strategy_pairs(races):
    """
    Find two drivers in the same race with IDENTICAL strategies.
    They should have IDENTICAL times and tie (or near-tie) in finishing order.
    This tells us if there's any noise/randomness.
    """
    print("\n=== IDENTICAL STRATEGY PAIRS ===")
    found = 0
    
    for r in races:
        if "finishing_positions" not in r:
            continue
        strats = list(r["strategies"].values())
        
        for i in range(len(strats)):
            for j in range(i+1, len(strats)):
                s1, s2 = strats[i], strats[j]
                if (s1["starting_tire"] == s2["starting_tire"] and
                    s1.get("pit_stops", []) == s2.get("pit_stops", [])):
                    actual = r["finishing_positions"]
                    p1 = actual.index(s1["driver_id"]) + 1
                    p2 = actual.index(s2["driver_id"]) + 1
                    print(f"  Race {r['race_id']}: {s1['driver_id']} P{p1} vs {s2['driver_id']} P{p2} - same strategy!")
                    found += 1
                    if found >= 5:
                        return

def find_single_compound_races(races):
    """
    Find races where a driver runs on ONE compound the whole race (no pit stops).
    Their finishing position relative to others tells us compound speed.
    """
    print("\n=== SINGLE-STINT DRIVERS (no pit stops) ===")
    
    no_stop_by_compound = defaultdict(list)
    
    for r in races:
        if "finishing_positions" not in r:
            continue
        actual = r["finishing_positions"]
        config = r["race_config"]
        
        for _, s in r["strategies"].items():
            if len(s.get("pit_stops", [])) == 0:
                did = s["driver_id"]
                pos = actual.index(did) + 1 if did in actual else None
                no_stop_by_compound[s["starting_tire"]].append({
                    "race": r["race_id"],
                    "pos": pos,
                    "laps": config["total_laps"],
                    "temp": config["track_temp"],
                    "base": config["base_lap_time"],
                })
    
    for compound, entries in no_stop_by_compound.items():
        avg_pos = sum(e["pos"] for e in entries if e["pos"]) / len(entries)
        print(f"\n{compound} no-stopper: {len(entries)} cases, avg finishing pos = {avg_pos:.1f}")
        for e in entries[:3]:
            print(f"  Race {e['race']}: P{e['pos']}, {e['laps']} laps, temp={e['temp']}")

def check_whether_model_fits_at_all(races):
    """
    For the top 10 races, try to find ANY parameter set that correctly
    predicts the winner. If we can't even predict the winner, the model
    structure is wrong.
    """
    print("\n=== CAN WE PREDICT THE WINNER? ===")
    
    def sim_time(s, config, so, ho, sd, md, hd, tr, tf):
        base, laps, pit = config["base_lap_time"], config["total_laps"], config["pit_lane_time"]
        tm = 1.0 + (config["track_temp"] - tr) * tf
        off = {"SOFT": so, "MEDIUM": 0.0, "HARD": ho}
        deg = {"SOFT": sd, "MEDIUM": md, "HARD": hd}
        pit_s = {p["lap"]: p["to_tire"] for p in s.get("pit_stops", [])}
        tire, age, t = s["starting_tire"], 0, 0.0
        for lap in range(1, laps + 1):
            age += 1
            t += base + off[tire] + age * deg[tire] * tm
            if lap in pit_s:
                t += pit
                tire, age = pit_s[lap], 0
        return t
    
    # Try a range of parameter sets and see which ones predict the winner correctly
    test_params = [
        (-1.5, 0.8, 0.10, 0.05, 0.02, 30.0, 0.02),   # our current
        (-0.5, 0.3, 0.20, 0.10, 0.04, 30.0, 0.02),   # calibrate best region
        (-0.3, 0.2, 0.30, 0.15, 0.06, 30.0, 0.01),   # more extreme
        (-0.1, 0.1, 0.40, 0.20, 0.08, 30.0, 0.00),   # very extreme
        (-2.0, 1.0, 0.05, 0.02, 0.01, 30.0, 0.00),   # original-ish
        # What if temp_factor is 0? (no temp effect)
        (-1.5, 0.8, 0.10, 0.05, 0.02, 30.0, 0.00),
        # What if degradation is much higher?
        (-1.5, 0.8, 0.50, 0.20, 0.05, 30.0, 0.00),
        # What if compound offset is tiny and degradation dominates?
        (-0.2, 0.1, 0.80, 0.40, 0.10, 30.0, 0.00),
    ]
    
    sample = [r for r in races[:50] if "finishing_positions" in r]
    
    print(f"\nTesting {len(test_params)} param sets on {len(sample)} races:")
    print(f"{'Params (so,ho,sd,md,hd,tr,tf)':>50} {'Winner%':>8} {'Exact%':>8}")
    print("-" * 70)
    
    for params in test_params:
        so, ho, sd, md, hd, tr, tf = params
        winner_correct = 0
        exact_correct = 0
        
        for r in sample:
            config = r["race_config"]
            times = {s["driver_id"]: sim_time(s, config, so, ho, sd, md, hd, tr, tf)
                     for s in r["strategies"].values()}
            predicted = sorted(times, key=times.__getitem__)
            actual = r["finishing_positions"]
            
            if predicted[0] == actual[0]:
                winner_correct += 1
            if predicted == actual:
                exact_correct += 1
        
        n = len(sample)
        label = f"so={so},ho={ho},sd={sd},md={md},hd={hd}"
        print(f"  {label:>48} {100*winner_correct/n:>7.1f}% {100*exact_correct/n:>7.1f}%")

def inspect_winner_strategies(races):
    """What do race winners' strategies look like?"""
    print("\n=== WINNER STRATEGY PATTERNS ===")
    
    winner_compounds = defaultdict(int)
    winner_stops = defaultdict(int)
    
    for r in races:
        if "finishing_positions" not in r:
            continue
        winner_id = r["finishing_positions"][0]
        for _, s in r["strategies"].items():
            if s["driver_id"] == winner_id:
                winner_compounds[s["starting_tire"]] += 1
                winner_stops[len(s.get("pit_stops", []))] += 1
                break
    
    total = sum(winner_compounds.values())
    print(f"\nWinner starting compound (n={total}):")
    for c, n in sorted(winner_compounds.items(), key=lambda x: -x[1]):
        print(f"  {c}: {n} ({100*n//total}%)")
    
    print(f"\nWinner number of pit stops:")
    for stops, n in sorted(winner_stops.items()):
        print(f"  {stops} stops: {n} ({100*n//total}%)")

def main():
    print("Loading races...")
    races = load_races(n_files=3)
    print(f"Loaded {len(races)} races")
    
    # Print first race in full detail
    print_race(races[0])
    print_race(races[1])
    
    # Check for identical strategy pairs (tests if model is deterministic)
    find_identical_strategy_pairs(races)
    
    # No-stop drivers (tests compound ordering)
    find_single_compound_races(races)
    
    # Winner patterns
    inspect_winner_strategies(races)
    
    # Can we predict the winner at all?
    check_whether_model_fits_at_all(races)

if __name__ == "__main__":
    main()
