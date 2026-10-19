# VLM4Coop

Reproducible extensions to read-only `../epymarl` and `../LVLM2P`. All Python,
installation, tests, training and inference run inside Docker. New artifacts stay
under this directory; the root Chinese `CODEX_CHECKPOINT.md` is the required
handoff exception. Preserve every existing run, checkpoint and raw response.

## Current status — 2026-10-19

- The fixed-500 MAPPO baseline completed 1,000,000 joint environment steps:
  `results/mappo_8x8_formal_fixed500_1000k_20261017_01`; validated statistics and
  curves are in its `_analysis` directory.
- Pure VLM formal evaluation `_04` completed 100 fixed episodes: 99 successes,
  3,102 joint steps and 3,150 HTTP attempts, with no probability cache.
  Its original Go endpoint and frozen evidence remain unchanged.
  Earlier `_01`, `_02`, `_03` attempts are incomplete and have no 100-episode score.
- Soft/3/seed0 completed 1,000,148 steps in
  `results/soft_8x8_fixed500_1000k_api_20261018_01`; offline validation and exports
  are in its `_analysis` directory.
- Hard/3/seed0 failed after its first 1000-step update in
  `results/hard_8x8_fixed500_1000k_api_20261019_01`, with 500 fixed training seeds,
  a 1,000k-step budget, cache-first teacher labels and a 1000 HTTP concurrency
  limit. It uses the same teacher probabilities as Soft and argmax CE for KD.
  OFF-control exhaustion during second-batch annotation stopped training at
  11:32 UTC+8; it has one completed evaluation, not a 1000k result. The container
  exited; its records remain preserved. See this run's INCIDENT_REPORT.md and
  incident_summary.json. Raw status counters partly retain the previous batch;
  future failure logging has been corrected and tested separately.
  The 4/5-agent runs are not included. The full research/paper is incomplete.
- The user subsequently clarified that returned reasoning is allowed when JSON
  is valid or numerically repairable. OFF remains the request preference; there
  are no reasoning-control retries. Hard `_02` restarted from seed 0 at 12:03
  UTC+8 and is running with a 1,000k-step budget, sharing the existing cache:
  `results/hard_8x8_fixed500_1000k_api_20261019_02`.
  Container: `vlm4coop_api_1792382610_cyc`; exec session 78486.
  One launch check passed without reading intermediate scores. This is a fresh
  initialization, not exact checkpoint continuation. See
  `reports/HARD_ACCEPT_REASONING_RUN.md`.
- Current implementation tests: 79 passed in CPU Docker, 63.12 s:
  `reports/pytest_accept_reasoning_20261019.log`. Synthetic teacher tests are not
  model results. Independent code and documented-invocation checks are saved in
  `reports/SOFT_API_1000K_CODE_REVIEW.md` and `SOFT_API_1000K_DOC_VALIDATION.md`.
  The Hard launch additionally passed two targeted Docker tests (7.19 s),
  cache-only engineering preflight and `HARD_API_1000K_CODE_REVIEW.md` /
  `HARD_API_1000K_DOC_VALIDATION.md`
  (same-family provisional).

The current detailed response policy and exact formal invocation are in
`reports/HARD_1000K_API_RUN.md`, `reports/SOFT_1000K_API_RUN.md` and
`reports/RESPONSE_FAILURE_FINAL_PROTOCOL.md`. Protocol decisions overriding the
original proposal are recorded in `reports/USER_PROTOCOL_DECISION.md`; resume
from the root checkpoint and verify actual files/containers first.

## Frozen student and environment protocol

All formal environments are 8×8, with 3/4/5 agents and the same number of foods.
The completed baseline and current Pure evaluation use 3 agents/3 foods. Native
EPyMARL MAPPO uses numerical Actor inputs, 10 complete episodes per rollout batch,
4 PPO epochs, training seed 0 and a 1,000k joint-environment-step budget. Student
evaluation samples actions rather than taking greedy actions. Complete batches
may cross a step budget; the actual count is logged.

Training initial-condition seeds are fixed to 0–499. An independent PCG64 sampler
(seed 0) draws 10 distinct seeds per batch; repetition across batches is allowed.
Evaluation uses seeds 1000000–1000099. Student training evaluates 100 episodes at
each crossed 1k-step threshold; Pure VLM evaluates these 100 seeds once. Reported
standard deviations are episode sample standard deviations (ddof=1), not
training-seed variation. Only successful episode lengths enter completion time.

## API teacher and response handling

The user replaced local Qwen with `deepseek-v4.1-flash` at
`https://opencode.ai/inference/openai/v1/chat/completions` for the new Soft run.
Completed Pure VLM used the frozen Go endpoint. Current requests use
`thinking.type=disabled`, omit `reasoning_effort` and omit `max_tokens`;
`top_p=0.95` is a requested parameter, not a measured provider-effective value.
The frozen v6 system/user prompts are in `src/vlm4coop/api_teacher.py`. One global
RGB includes stable zero-based food IDs F0..F(N-1), Agent IDs and levels; actual
state coordinates are not provided as text. The image is a deterministic PIL
render of the real LBF state, not the original pyglet window or an AI image.
Students do not receive RGB.

Responses may include visible brief reasoning before the final complete JSON.
Require exactly all Agent IDs and six finite numeric action values per row.
Pure VLM requires probabilities in [0,1] summing to one; after three failed
format attempts it executes NONE for every Agent for that state. Successful
labels execute per-agent argmax, without an external planner. Each effective
state makes fresh requests even when an image repeats.

VLM4Coop clips each valid-schema row to [0,1] then normalizes. All-zero clipped
rows remain invalid. After three format failures, `teacher_valid=0` masks only
KD for that state; all native PPO/Critic updates remain. No student-probability
fallback is used. Failed labels/default actions do not enter the probability
cache. See `reports/KD_MASK_IMPLEMENTATION.md` for formulas and tested scope.
Training's teacher request limit is 1000; Pure overrides it to 100. Training
uses cache-first raw-response revalidation; Pure does not open SQLite. The
cache key includes top_p, model, generation settings, both prompts and renderer.
It excludes concurrency, pure_vlm_use_cache, timeout_seconds, parser_version,
failure_policy and api_control_version, as requested. Valid labels are reused
for all four PPO epochs. Student evaluation makes no teacher requests.

OFF is a request preference. Responses containing reasoning are accepted and
cached on the first valid or numerically repairable all-agent JSON, without
reasoning-control retries. Final `content` takes priority; when empty, JSON can
be extracted from `reasoning_content`. Malformed final content cannot be replaced
by an earlier draft. Returned reasoning is recorded as `returned_reasoning`;
missing usage stays null with `thinking_evidence=not_reported`, and explicit zero
is `reported_zero`. Never describe missing telemetry as proof of zero internal
reasoning. Completed Soft used the prior stricter acceptance policy; this
protocol difference must be disclosed in comparisons. Retries for HTTP transport
are audited separately from the three format attempts; quota/auth failures do
not become NONE or KD masks.

## Docker development and evidence

Reuse the existing pinned runtime; do not install Python packages on the host.
Run tests with:

```bash
VLM4Coop/scripts/docker_cpu.sh python -m pytest -q VLM4Coop/tests -o cache_dir=VLM4Coop/runtime/pytest_cache
```

CPU/API wrappers use the current UID/GID, 4 CPUs/8 GiB, no GPU/ports, read-only
project root and writable VLM4Coop mount. API credentials enter through echo-off
stdin only; never store them in configuration, `.env`, command arguments or logs.
The historical local-Qwen GPU scripts use GPU 0 only under previous authorization;
they do not define the current API protocol.

Runs retain source/config/dependency snapshots, raw responses and usage, image
frames, action/episode logs, timing and cache counters. Final Pure analysis runs
offline in Docker using `scripts/analyze_pure_vlm.py`; incomplete attempts do not
produce full-evaluation scores. MAPPO evidence and protocol: `reports/FIXED500_RUN.md`.
Historical local/API probes and unrestricted-scenario baselines are preserved;
do not mix them with the current comparison protocol.
