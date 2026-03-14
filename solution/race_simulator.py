#!/usr/bin/env python3
import json, sys

COMPOUND_OFFSET = {"SOFT": -1.5, "MEDIUM": 0.0, "HARD": 0.8}
DEG_RATE        = {"SOFT":  0.10, "MEDIUM": 0.05, "HARD": 0.02}
TEMP_REF        = 30.0
TEMP_FACTOR     = 0.02

def compute_driver_total_time(strategy, config):
    base      = config["base_lap_time"]
    laps      = config["total_laps"]
    pit_time  = config["pit_lane_time"]
    temp_mult = 1.0 + (config["track_temp"] - TEMP_REF) * TEMP_FACTOR
    pit_sched = {p["lap"]: p["to_tire"] for p in strategy.get("pit_stops", [])}
    tire, age, total = strategy["starting_tire"], 0, 0.0
    for lap in range(1, laps + 1):
        age   += 1
        total += base + COMPOUND_OFFSET[tire] + age * DEG_RATE[tire] * temp_mult
        if lap in pit_sched:
            total += pit_time
            tire, age = pit_sched[lap], 0
    return total

def main():
    data   = json.loads(sys.stdin.read())
    config = data["race_config"]
    times  = {s["driver_id"]: compute_driver_total_time(s, config)
              for s in data["strategies"].values()}
    print(json.dumps({
        "race_id":             data["race_id"],
        "finishing_positions": sorted(times, key=times.__getitem__)
    }))

if __name__ == "__main__":
    main()
