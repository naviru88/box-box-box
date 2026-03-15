#!/usr/bin/env python3
"""
Direct parameter extraction from historical race data.
Run from repo root: python3 solution/extract_params.py
"""
import json, glob, sys, math
from collections import defaultdict

DATA_DIR = "data/historical_races"

def load_races(n_files=5):
    races = []
    for f in sorted(glob.glob(f"{DATA_DIR}/*.json"))[:n_files]:
        data = json.load(open(f))
        races.extend(data if isinstance(data, list) else [data])
    return races

def sim(strategy, config, soft_off, hard_off, soft_deg, med_deg, hard_deg, temp_ref, temp_fac):
    base, laps, pit_time = config["base_lap_time"], config["total_laps"], config["pit_lane_time"]
    tm = 1.0 + (config["track_temp"] - temp_ref) * temp_fac
    offsets = {"SOFT": soft_off, "MEDIUM": 0.0, "HARD": hard_off}
    degs    = {"SOFT": soft_deg, "MEDIUM": med_deg, "HARD": hard_deg}
    pit_s   = {p["lap"]: p["to_tire"] for p in strategy.get("pit_stops", [])}
    tire, age, t = strategy["starting_tire"], 0, 0.0
    for lap in range(1, laps + 1):
        age += 1
        t += base + offsets[tire] + age * degs[tire] * tm
        if lap in pit_s:
            t += pit_time
            tire, age = pit_s[lap], 0
    return t

def score(races, soft_off, hard_off, soft_deg, med_deg, hard_deg, temp_ref, temp_fac):
    correct = 0
    for r in races:
        if "finishing_positions" not in r:
            continue
        times = {s["driver_id"]: sim(s, r["race_config"], soft_off, hard_off, soft_deg, med_deg, hard_deg, temp_ref, temp_fac)
                 for s in r["strategies"].values()}
        if sorted(times, key=times.__getitem__) == r["finishing_positions"]:
            correct += 1
    return correct

def analyze_structure(races):
    """Look at what strategies actually appear to understand the scale of differences."""
    print("\n=== RACE STRUCTURE ANALYSIS ===")
    
    # For races won by different starting compounds, what's the margin?
    stint_lengths = defaultdict(list)
    
    for r in races[:500]:
        if "finishing_positions" not in r:
            continue
        config = r["race_config"]
        total_laps = config["total_laps"]
        winner_id = r["finishing_positions"][0]
        
        for _, s in r["strategies"].items():
            pits = s.get("pit_stops", [])
            # Record stint lengths per compound
            tire = s["starting_tire"]
            prev_lap = 0
            for pit in pits:
                stint_lengths[tire].append(pit["lap"] - prev_lap)
                prev_lap = pit["lap"]
                tire = pit["to_tire"]
            stint_lengths[tire].append(total_laps - prev_lap)
    
    print("\nTypical stint lengths by compound:")
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        lengths = stint_lengths[compound]
        if lengths:
            print(f"  {compound}: avg={sum(lengths)/len(lengths):.1f} laps, "
                  f"min={min(lengths)}, max={max(lengths)}, n={len(lengths)}")

def find_params_by_isolating(races):
    """
    Find params by looking at races where only ONE variable differs between drivers.
    
    Key insight: If driver A and B have IDENTICAL strategies except A pits 1 lap earlier,
    the time difference tells us the marginal value of 1 lap of tire age.
    """
    print("\n=== ISOLATING DEGRADATION RATES ===")
    
    # Find pairs of drivers with same compounds, same number of stops, 
    # but pit stop lap differs by exactly 1
    isolation_pairs = []
    
    for r in races[:1000]:
        if "finishing_positions" not in r:
            continue
        config = r["race_config"]
        actual = r["finishing_positions"]
        strats = list(r["strategies"].values())
        
        for i in range(len(strats)):
            for j in range(i+1, len(strats)):
                s1, s2 = strats[i], strats[j]
                p1 = s1.get("pit_stops", [])
                p2 = s2.get("pit_stops", [])
                
                # Same number of stops
                if len(p1) != len(p2) or len(p1) == 0:
                    continue
                
                # Same compounds throughout
                same_start = s1["starting_tire"] == s2["starting_tire"]
                same_compounds = all(p1[k]["to_tire"] == p2[k]["to_tire"] for k in range(len(p1)))
                
                if not same_start or not same_compounds:
                    continue
                
                # All pit laps the same except one differs by small amount
                lap_diffs = [abs(p1[k]["lap"] - p2[k]["lap"]) for k in range(len(p1))]
                if sum(lap_diffs) > 0 and sum(lap_diffs) <= 5 and max(lap_diffs) <= 5:
                    d1, d2 = s1["driver_id"], s2["driver_id"]
                    pos1 = actual.index(d1) + 1 if d1 in actual else None
                    pos2 = actual.index(d2) + 1 if d2 in actual else None
                    
                    if pos1 and pos2:
                        # Who pitted earlier?
                        earlier_pitter = d1 if p1[0]["lap"] < p2[0]["lap"] else d2
                        earlier_pos = pos1 if earlier_pitter == d1 else pos2
                        isolation_pairs.append({
                            "compound": s1["starting_tire"],
                            "lap_diff": p2[0]["lap"] - p1[0]["lap"],  # positive = d2 pits later
                            "pos_diff": pos2 - pos1,  # positive = d2 finishes worse
                            "earlier_wins": earlier_pos < (pos2 if earlier_pitter == d1 else pos1),
                            "total_laps": config["total_laps"],
                        })
    
    print(f"Found {len(isolation_pairs)} isolation pairs")
    
    # When does pitting EARLIER win vs LATER?
    by_compound = defaultdict(lambda: {"earlier_wins": 0, "later_wins": 0, "lap_diffs": []})
    for pair in isolation_pairs:
        c = pair["compound"]
        by_compound[c]["lap_diffs"].append(pair["lap_diff"])
        if pair["earlier_wins"]:
            by_compound[c]["earlier_wins"] += 1
        else:
            by_compound[c]["later_wins"] += 1
    
    for compound, data in by_compound.items():
        total = data["earlier_wins"] + data["later_wins"]
        print(f"\n  {compound}: earlier_wins={data['earlier_wins']}, later_wins={data['later_wins']}")
        print(f"    Earlier wins {100*data['earlier_wins']//total}% of the time")

def exhaustive_fine_search(races, n_races=500):
    """
    Based on calibrate.py results showing best region around:
    soft_off~-1.0, hard_off~0.5, soft_deg~0.2, med_deg~0.08, hard_deg~0.04
    
    Do a much finer search in that region AND expand outward.
    """
    print("\n=== FINE-GRAINED SEARCH (expanded range) ===")
    
    sample = [r for r in races[:n_races] if "finishing_positions" in r]
    print(f"Using {len(sample)} races")
    
    best_score = -1
    best_params = None
    
    # Expanded ranges based on calibrate.py trends (loss kept improving at boundaries)
    import itertools
    
    ranges = {
        "soft_off":  [-0.8, -0.6, -0.4, -0.3, -0.2, -0.1],
        "hard_off":  [0.1, 0.2, 0.3, 0.4, 0.5],
        "soft_deg":  [0.20, 0.25, 0.30, 0.35, 0.40],
        "med_deg":   [0.08, 0.10, 0.12, 0.15, 0.18],
        "hard_deg":  [0.03, 0.04, 0.05, 0.06, 0.08],
        "temp_ref":  [25.0, 30.0],
        "temp_fac":  [0.00, 0.01, 0.02, 0.03],
    }
    
    total = 1
    for v in ranges.values():
        total *= len(v)
    print(f"Testing {total} combinations...")
    
    n = 0
    for so in ranges["soft_off"]:
        for ho in ranges["hard_off"]:
            for sd in ranges["soft_deg"]:
                for md in ranges["med_deg"]:
                    for hd in ranges["hard_deg"]:
                        if not (sd > md > hd):
                            continue
                        for tr in ranges["temp_ref"]:
                            for tf in ranges["temp_fac"]:
                                n += 1
                                s = score(sample, so, ho, sd, md, hd, tr, tf)
                                if s > best_score:
                                    best_score = s
                                    best_params = (so, ho, sd, md, hd, tr, tf)
                                    pct = 100.0 * s / len(sample)
                                    print(f"  [{n}] {pct:.1f}% ({s}/{len(sample)}) "
                                          f"so={so} ho={ho} sd={sd} md={md} hd={hd} tr={tr} tf={tf}")
    
    return best_params, best_score

def write_params(so, ho, sd, md, hd, tr, tf):
    """Update race_simulator.py with new parameters."""
    content = open("solution/race_simulator.py").read()
    
    import re
    content = re.sub(r'"SOFT":\s*[-\d.]+,(\s*#[^\n]*)?(\s*"MEDIUM")',
                     f'"SOFT": {so},\\2', content, count=1)
    content = re.sub(r'"HARD":\s*[-\d.]+,(\s*#[^\n]*)?(\s*\})\s*\nDEG',
                     f'"HARD": {ho},\\2\nDEG', content, count=1)
    
    content = re.sub(r'(DEG_RATE[^}]+"SOFT":\s*)[\d.]+',  f'\\g<1>{sd}', content)
    content = re.sub(r'(DEG_RATE[^}]+"MEDIUM":\s*)[\d.]+', f'\\g<1>{md}', content)
    content = re.sub(r'(DEG_RATE[^}]+"HARD":\s*)[\d.]+',   f'\\g<1>{hd}', content)
    content = re.sub(r'TEMP_REF\s*=\s*[\d.]+',  f'TEMP_REF    = {tr}', content)
    content = re.sub(r'TEMP_FACTOR\s*=\s*[\d.]+', f'TEMP_FACTOR = {tf}', content)
    
    open("solution/race_simulator.py", "w").write(content)
    print(f"\nUpdated solution/race_simulator.py:")
    print(f"  COMPOUND_OFFSET SOFT={so}, HARD={ho}")
    print(f"  DEG_RATE SOFT={sd}, MEDIUM={md}, HARD={hd}")
    print(f"  TEMP_REF={tr}, TEMP_FACTOR={tf}")

def main():
    print("Loading historical races...")
    races = load_races(n_files=10)
    print(f"Loaded {len(races)} races")
    
    # First understand the data structure
    analyze_structure(races)
    
    # Find isolation pairs to understand degradation direction
    find_params_by_isolating(races)
    
    # Fine search in the region where calibrate.py was converging
    best_params, best_score = exhaustive_fine_search(races, n_races=1000)
    
    if best_params:
        so, ho, sd, md, hd, tr, tf = best_params
        pct = 100.0 * best_score / min(1000, len(races))
        print(f"\nBest found: {pct:.1f}% accuracy")
        print(f"Params: soft_off={so}, hard_off={ho}, soft_deg={sd}, med_deg={md}, hard_deg={hd}, temp_ref={tr}, temp_fac={tf}")
        write_params(so, ho, sd, md, hd, tr, tf)
    
    print("\nDone! Run ./test_runner.sh to evaluate.")

if __name__ == "__main__":
    main()
