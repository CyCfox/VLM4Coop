# Active command entry points

Python implementations live in `src/vlm4coop/cli/`. Use canonical module paths
or categorized script paths; flat compatibility wrappers have been moved intact
to `old/` and must not be used for new runs or reproductions.

- `analysis/`: statistics and exports.
- `evaluation/`: the existing explicit formal-suite command; never auto-run.
- `validation/`: raw-log/output/cache auditing and offline replay.
- `probes/`: explicitly invoked integration/model probes.
- `diagnostics/`: environment/runtime checks and launch records.
- `old/`: retired flat wrappers, bytecode and older source/config patch scripts.

Actual `docker_*.sh`, `start_*.sh` and finish payloads remain here and call the
canonical modules internally. Their GPU/resource/credential behavior is unchanged.
All Python execution must use Docker. See `../docs/ENTRYPOINTS.md` for current
paths. No experiments were launched during this migration.
