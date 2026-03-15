#!/usr/bin/env python3
import json, sys

COMPOUND_OFFSET = {"SOFT": -15.0, "MEDIUM": 0.0, "HARD": 11.0}
DEG_RATE        = {"SOFT": 0.5,   "MEDIUM": 0.08, "HARD": 0.02}
TEMP_REF        = 30.0
TEMP_FACTOR     = 0.0

def simulate_race(race_input):
    c    = race_input["race_config"]
    base = c["base_lap_time"]
    laps = c["total_laps"]
    pit  = c["pit_lane_time"]
    tm   = 1.0 + (c["track_temp"] - TEMP_REF) * TEMP_FACTOR
    times = {}
    grid  = {}
    for pos_key, s in race_input["strategies"].items():
        did       = s["driver_id"]
        grid[did] = int(pos_key[3:])
        ps        = {p["lap"]: p["to_tire"] for p in s.get("pit_stops", [])}
        tire, age, t = s["starting_tire"], 0, 0.0
        for lap in range(1, laps + 1):
            age += 1
            t   += base + COMPOUND_OFFSET[tire] + age * DEG_RATE[tire] * tm
            if lap in ps:
                t   += pit
                tire = ps[lap]
                age  = 0
        times[did] = t
    return {
        "race_id":             race_input["race_id"],
        "finishing_positions": sorted(times, key=lambda d: (times[d], grid[d]))
    }

def main():
    print(json.dumps(simulate_race(json.loads(sys.stdin.read()))))

if __name__ == "__main__":
    main()
