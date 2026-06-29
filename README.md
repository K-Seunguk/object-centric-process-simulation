# OCPS-VF Analysis and Simulation

This folder is a GitHub-ready copy of the OCPS-VF workflow. It keeps the source
OCEL data, the analysis code used to build simulation parameters, the simulation
runtime, generated simulation inputs/outputs, and evaluation assets.

## Structure

- `data/`: source OCEL JSON logs for `om`, `p2p`, and `logistics`
- `object_analyzer.py`, `process_discoverer.py`, `performance_analyzer.py`,
  `decision_point_analyzer.py`: analysis components used in memory by the
  simulation input builder
- `simulation_input_builder.py`: builds simulation input JSON files from the
  analysis pipeline
- `run_pipeline.py`: runs OCEL-to-input-to-simulation in one command, without
  writing intermediate analysis artifacts
- `simulation/input/`: generated simulation input JSON files
- `simulation/src/`: simulation runtime
- `simulation/output/`: generated simulated OCEL JSON logs
- `eval/`: paper/reference material and evaluation results

## Setup

```bash
pip install -r requirements.txt
```

The original experiments were run with Python 3.11/3.12 and `pm4py` + `simpy`.

## Run Full Pipeline

Run from this directory:

```bash
python run_pipeline.py logistics   --case-count 2000   --seed 7   --start-time 2023-04-03T10:00:00+00:00
```

Use `all` instead of a dataset name to run `om`, `p2p`, and `logistics`.

## Rebuild Simulation Inputs Only

```bash
python simulation_input_builder.py all --output-dir simulation/input
```

The current performance extraction includes interarrival statistics for every
`INITIATE` object type. Top-level initiate types are used for root case arrivals;
non-top initiate types are inserted as deferred arrivals according to the object
graph cardinalities and their own interarrival distributions.

## Run Simulation From Existing Inputs

Example:

```bash
python -m simulation.src.runner   --sim-input simulation/input/simulation_input_logistics.json   --out simulation/output/simulated_logistics.json   --start-time 2040-01-01T00:00:00+00:00   --seed 7   --case-count 2000   --progress-every 100
```

## Evaluate

```bash
python evaluation.py
```

The evaluator compares original logs under `data/` against simulated logs under
`simulation/output/` and writes results to `eval/evaluation_results.json`.
