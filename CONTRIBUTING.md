# Contributing

Thanks for helping people understand each other.

1. `python -m pytest` must pass, and `hermes plugins validate .` must report *Validation passed*.
2. Keep the plugin side (`live_translate/`) free of heavy dependencies — it runs inside Hermes. Model code belongs in
   `lt_engine/`, which has its own environment.
3. New voice or recognition backends: add a spec to `lt_engine/voices.py` or `lt_engine/asr.py`, or ship a worker
   (`lt_engine/worker.py` protocol) so it can live in its own environment.
4. Measure before claiming speed or quality: run the engine on a real recording
   (`python -m lt_engine --to es --file talk.wav --no-play`) and include the `summary` line it prints.
5. No telemetry. Anything that talks to the network must be opt-in and listed in the README's disclosures.
