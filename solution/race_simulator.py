#!/usr/bin/env python3
"""
F1 Race Simulator - Box Box Box Challenge
=========================================

Usage:
    cat data/test_cases/inputs/test_001.json | python solution/race_simulator.py

Tire Model (calibrate parameters by running: python analysis/calibrate.py)
---------------------------------------------------------------------------
Per-lap formula:
    tire_age += 1                                   # increments BEFORE lap time calc
    lap_time  = base_lap_time
              + COMPOUND_OFFSET[compound]           # speed offset vs MEDIUM reference
              + tire_age * DEG_RATE[compound] * temp_mult   # degradation

    temp_mult = 1.0 + (track_temp - TEMP_REF) * TEMP_FACTOR

Total race time = sum(lap_times) + n_pit_stops * pit_lane_time
Winner          = driver with lowest total race time

Key rule from regulations.md:
    "At the start of each lap, tire age increments by 1 BEFORE calculating lap time"
    => First lap on fresh tires uses tire_age = 1 (NOT 0)
"""

import json
import sys

# ---------------------------------------------------------------------------
# TIRE MODEL PARAMETERS
# Default values below are reasonable starting guesses.
# Run:  python analysis/calibrate.py
# to optimise these against the 30,000 historical races and auto-update them.
# ---------------------------------------------------------------------------

# Speed offset per compound relative to MEDIUM (seconds per lap, fresh tires)
# SOFT is faster (negative), HARD is slower (positive)
COMPOUND_OFFSET = {
    "SOFT":   -1.5,
    "MEDIUM":  0.0,   # reference point
    "HARD":    0.8,
}

# Linear degradation: seconds added per lap of tire age
# Ordering must hold: SOFT_DEG > MEDIUM_DEG > HARD_DEG
DEG_RATE = {
    "SOFT":   0.10,
    "MEDIUM": 0.05,
    "HARD":   0.02,
}

# Temperature model — higher track temp → more degradation
TEMP_REF    = 30.0   # °C at which temp_mult == 1.0
TEMP_FACTOR = 0.02   # fractional change in degradation per °C delta


# ---------------------------------------------------------------------------
# SIMULATION CORE
# ---------------------------------------------------------------------------

def compute_driver_total_time(strategy: dict, config: dict) -> float:
    """
    Simulate one driver's race and return their total elapsed time (seconds).

    Tire age rule (from regulations.md):
        tire_age increments by 1 at the START of each lap, before the lap
        time is calculated.  Fresh tires therefore produce age=1 on lap 1.

    Pit stop rule:
        A pit stop listed for lap N is served at the END of lap N.
        The pit_lane_time penalty is added to total time.
        tire_age resets to 0 after the stop (so the next lap starts at age=1).
    """
    base_lap_time = config["base_lap_time"]
    total_laps    = config["total_laps"]
    pit_lane_time = config["pit_lane_time"]
    track_temp    = config["track_temp"]

    temp_mult = 1.0 + (track_temp - TEMP_REF) * TEMP_FACTOR

    # Map lap number → new compound fitted at end of that lap
    pit_schedule = {pit["lap"]: pit["to_tire"]
                    for pit in strategy.get("pit_stops", [])}

    current_tire = strategy["starting_tire"]
    tire_age     = 0       # resets to 0 after each stop; incremented before use
    total_time   = 0.0

    for lap in range(1, total_laps + 1):

        # ── Tire age increments BEFORE lap time calculation (regulations §Timing) ──
        tire_age += 1

        lap_time = (
            base_lap_time
            + COMPOUND_OFFSET[current_tire]
            + tire_age * DEG_RATE[current_tire] * temp_mult
        )
        total_time += lap_time

        # ── Pit stop served at END of this lap ──
        if lap in pit_schedule:
            total_time  += pit_lane_time
            current_tire = pit_schedule[lap]
            tire_age     = 0   # fresh set; next lap starts at age=1

    return total_time


def simulate_race(race_input: dict) -> dict:
    """Return {'race_id': ..., 'finishing_positions': [ordered driver IDs]}."""
    race_id    = race_input["race_id"]
    config     = race_input["race_config"]
    strategies = race_input["strategies"]

    driver_times = {
        strat["driver_id"]: compute_driver_total_time(strat, config)
        for strat in strategies.values()
    }

    finishing_positions = sorted(driver_times, key=driver_times.__getitem__)

    return {
        "race_id":             race_id,
        "finishing_positions": finishing_positions,
    }


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    race_input = json.loads(sys.stdin.read())
    result     = simulate_race(race_input)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
