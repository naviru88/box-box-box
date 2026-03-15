# F1 Race Simulator Solution

## Overview

This solution simulates F1 races to predict finishing positions based on tire strategies and track conditions. The core insight is that race outcomes are **fully deterministic** — the winner is simply the driver with the lowest total race time.

## Solution Architecture

```
solution/
  simulator.py        ← Main solution (reads stdin → writes stdout)
  run_command.txt     ← Command: "python3 solution/simulator.py"

analysis/
  analyze.py          ← Initial data exploration and grid search
  calibrate.py        ← Full parameter optimization (run once on data)
  validate.py         ← Accuracy validation against test/historical data
  deep_analysis.py    ← Advanced analysis (natural experiments, scipy)
  optimized_params.json ← Best params found by calibration
```

## The Model

### Lap Time Formula

```
lap_time = base_lap_time 
         + COMPOUND_OFFSET[compound] 
         + (tire_age × DEG_RATE[compound] × temp_multiplier)

temp_multiplier = 1 + (track_temp - TEMP_REF) × TEMP_FACTOR
tire_age = laps completed on current tire set (starts at 0)
```

### Total Race Time

```
total_time = Σ(lap_times) + Σ(pit_lane_time per pit stop)
```

### Parameters (to be calibrated)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `COMPOUND_OFFSET["SOFT"]` | -1.5 | SOFT is ~1.5s faster than MEDIUM per lap (fresh) |
| `COMPOUND_OFFSET["MEDIUM"]` | 0.0 | Reference compound |
| `COMPOUND_OFFSET["HARD"]` | 0.8 | HARD is ~0.8s slower than MEDIUM per lap (fresh) |
| `DEG_RATE["SOFT"]` | 0.10 | SOFT loses 0.10s/lap with tire age |
| `DEG_RATE["MEDIUM"]` | 0.05 | MEDIUM loses 0.05s/lap |
| `DEG_RATE["HARD"]` | 0.02 | HARD loses 0.02s/lap |
| `TEMP_REF` | 30.0 | Reference temperature (°C) |
| `TEMP_FACTOR` | 0.02 | 2% more degradation per °C above reference |

## Quick Start

### Step 1: Calibrate Parameters (IMPORTANT — run first!)

```bash
python3 analysis/calibrate.py
```

This loads the historical race data and finds the optimal parameters for the tire model. It automatically updates `solution/simulator.py` with the best values.

### Step 2: Validate

```bash
python3 analysis/validate.py
```

Checks accuracy on:
- Historical race data (should see 95%+ after calibration)
- Test cases (the 100 official cases)

### Step 3: Run Test Suite

```bash
./test_runner.sh
```

### Step 4: Debug Failures (if needed)

```bash
python3 analysis/validate.py --debug TEST_001
```

Shows per-driver times and the first divergence from expected results.

## How Calibration Works

### If raw lap time data is available in historical records:
- Direct linear regression on (tire_age, lap_time) pairs per compound
- Extracts exact offset and degradation rate for each compound
- Temperature effects fit separately

### If only finishing positions are available:
- **Pairwise ranking loss**: for each consecutive pair in actual order,
  penalize if our model predicts wrong order
- **Grid search** over parameter space (coarse → fine)
- **scipy differential_evolution** for global optimization
- Uses up to 30,000 races × 19 pairs = ~570k ordering constraints

## Key Insights

1. **Deterministic**: No randomness — same inputs always produce same output
2. **Independent drivers**: Cars don't interact (no overtaking physics)
3. **Linear degradation**: Tire performance degrades linearly with tire age
4. **Temperature scales degradation**: Higher temp → faster deg for all compounds
5. **SOFT crossover**: SOFT starts fastest but crossovers slower compounds after N laps
   - Crossover point where SOFT equals MEDIUM: `(-soft_offset) / (soft_deg - med_deg)`

## Crossover Analysis

At what tire age does SOFT become slower than MEDIUM?

```
SOFT_time(age) = base + soft_offset + age × soft_deg
MEDIUM_time(age) = base + 0 + age × med_deg

Crossover at: age = -soft_offset / (soft_deg - med_deg)
           = 1.5 / (0.10 - 0.05) = 30 laps (at reference temp)
```

This explains the optimal pit strategy: pit SOFT before lap ~30.

## Troubleshooting

**Low accuracy (<50%)?**
- Run `python3 analysis/calibrate.py` first
- Check if `TEMP_FACTOR` should be 0.0 (temperature might not affect things)

**~50% accuracy?**
- Model gets relative order wrong between similar-strategy drivers
- Fine-tune degradation rates more carefully

**90%+ but not perfect?**
- Check for non-linear degradation (quadratic term?)
- Check if pit stop time includes an in-lap/out-lap penalty
- Check if temperature affects compound offset (not just deg rate)
