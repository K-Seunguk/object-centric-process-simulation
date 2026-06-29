# Simulation Model

This package is the SimPy-oriented OCPS simulator. It is self-contained inside
`pm/ocps/simulation_model` and no longer imports the legacy
`pm/ocps/simulation/dispatcher.py`.

## Components

- `configuration_manager.py`: loads and validates simulation input.
- `process_model_builder.py`: builds the object-aware Petri net from input places, transitions, and arcs.
- `object_graph_store.py`: stores and queries case-scoped object graph relationships.
- `object_graph_generator.py`: prepares arrivals, deferred arrivals, and initial marking.
- `object_state_store.py`: owns token, marking, completion, and case lifecycle state.
- `resource_scheduler.py`: handles resource requirements and busy intervals used by the engine.
- `synchronization_controller.py`: selects enabled transitions, token-consumption sets, and branch choices.
- `ocel_logger.py`: builds OCEL 2.0 output and event/object relationship payloads.
- `runtime_context.py`: shared runtime helpers used by the SimPy engine and component handlers.
- `transition_executor.py`: applies completed transition results to object state and OCEL logs.
- `simulation_engine.py`: SimPy environment, arrival processes, transition processes, and scheduling.
- `runner.py`: command-line entry point.

## Run

```bash
/home/ksu_aim25/miniconda3/envs/pm_ocps/bin/python -m pm.ocps_vf.simulation.src.runner \
  --sim-input pm/ocps_vf/simulation/input/simulation_input_om.json \
  --out pm/ocps_vf/simulation/output/simulated_om.json \
  --start-time 2023-04-03T00:00:00+00:00 \
  --seed 42 \
  --case-count 2000 \
  --progress-every 10 \
  --heartbeat-seconds 10

/home/ksu_aim25/miniconda3/envs/pm_ocps/bin/python -m pm.ocps_vf.simulation.src.runner \
  --sim-input pm/ocps_vf/simulation/input/simulation_input_p2p.json \
  --out pm/ocps_vf/simulation/output/simulated_p2p.json \
  --start-time 2022-04-01T00:00:00+00:00 \
  --seed 42 \
  --case-count 927 \
  --progress-every 10 \
  --heartbeat-seconds 10

/home/ksu_aim25/miniconda3/envs/pm_ocps/bin/python -m pm.ocps_vf.simulation.src.runner \
  --sim-input pm/ocps_vf/simulation/input/simulation_input_logistics.json \
  --out pm/ocps_vf/simulation/output/simulated_logistics.json \
  --start-time 2023-05-22T00:00:00+00:00 \
  --seed 42 \
  --case-count 600 \
  --progress-every 10 \
  --heartbeat-seconds 10
```
