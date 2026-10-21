# VLM4Coop

Reproducible extensions to read-only `../epymarl` and `../LVLM2P`. All Python,
installation, tests, training and inference run inside Docker. New artifacts stay
under this directory; the root Chinese `CODEX_CHECKPOINT.md` is the required
handoff exception. Preserve every existing run, checkpoint and raw response.

## Current status — 2026-10-20

The authorized MPE VLM4Coop Soft formal run is now running in
`results/soft_mpe_spread_fixed500_1000k_api_20261020_01`:1Mstep budget,
40complete25step environments/batch,4PPO epochs and100fixed-scene evaluation
every1000steps,500training scenes,API concurrency/guard ceiling1000, OFF and
cache-first. Existing KD schedule disables teacher queries and KD at update
step100000 (last annotated update99000); MAPPO continues afterward.
Five Docker regressions and one8state real-API/cache integration preflight
passed. Only startup metadata was checked; completion is not assumed.
[Protocol and execution evidence](reports/mpe_soft1000_20261020/VALIDATION.md).
The same container will validate raw records and automatically export curves.
At the user-requested runtime check, it was still awaiting the first teacher
batch:40episodes sampled,0PPO/evaluations;976accepted labels persisted,
18HTTP429 records and16network-wait threads. Missing evaluations.jsonl at
that point did not indicate container exit. See the validation report snapshot.

The user-authorized MPE MAPPO40 rerun completed and passed raw-log validation:
`results/mappo_mpe_spread_fixed500_1000k_batch40_20261020_03`. It trained exactly
1,000,000 joint steps (40,000 complete episodes), using40parallel environments
and25steps per episode, with4PPO epochs and100fixed-scene evaluation every1000
training steps. All1000 evaluations were exported, with seed-pool replay and
train/test isolation verified. Final native team return is−57.79±15.21
(100episodes, sampleSD); final all-three coverage is0/100. Automatic
[learning curves](results/mappo_mpe_spread_fixed500_1000k_batch40_20261020_03_analysis/learning_curves.png)
and CSV/PDF/SVG are available. [Execution and validation](reports/mappo_mpe_batch40_20261020/VALIDATION.md).

The MPE Pure VLM formal evaluation also completed and passed native/RGB replay:
`results/pure_vlm_mpe_100_uncached_20261020_01`,100fixed scenes×25steps,
100configured HTTP concurrency,2500decisions/2514HTTP attempts, no probability
cache and reported reasoning tokens0. Native return is−46.28±14.84 (episode
sampleSD), final all-three coverage0/100. [Verified report](results/pure_vlm_mpe_100_uncached_20261020_01/REPORT.md).
The new Soft run above is active; MAPPO40 and Pure100 remain completed.
Dedicated configs authorize only their respective methods; the base config
remains paused. Prior runs and their frozen protocols are preserved.

The current OFF teacher completed the authorized3environment×25step uncached
capability probe in78.86s. All75HTTP responses were200, contained valid labels
without retries, and reported0reasoning tokens. Final covered landmarks were
0/3,0/3,1/3; final and ever-all-covered success were both0/3episodes. Native
return was−38.80±10.54 (episode sampleSD,n=3). All75native steps/RGB inputs were
replayed offline. The teacher approached goals but still misread some velocity
directions and coverage. See [actual capability report](results/mpe_teacher_off_capability_20261020_01/analysis/CAPABILITY_REPORT.md).
This is a small probe; it is separate from the completed formal runs above.

The user's exact MPE teacher system/user prompts are installed. One authorized
uncached capability probe completed: three parallel environments,25steps each,
seeds1000000..1000002 in `results/mpe_teacher_user_prompt_real_20261020_01`.
It returned reasoning in74/75 requests despite requestedOFF. The new template
OFF control passed6/6 real image calls with reported0 reasoning tokens.
No formal student/teacher training was resumed. See
[probe protocol](reports/mpe_teacher_probe_20261020/PROTOCOL.md).

The MPE formal standard is now:
after each forty complete25-step episodes (1000joint training steps), perform one
four-epoch PPO update; evaluate100fixed scenarios every1000training steps.
See [current protocol](reports/mpe40_fixed_off_20261020/PROTOCOL.md).
Future MPE completion exports automatically include PNG/PDF/SVG learning curves,
their dataCSV, caption and plotting-version metadata. The original completed
run's curves are now [available here](results/mappo_mpe_spread_fixed500_1000k_20261020_01_analysis/plots_v2/learning_curves.png).
The earlier implementation passed121Docker regressions and the final layout
check; [historical validation](reports/mpe_batch250_20261020/VALIDATION.md).
The eval250 rerun in `results/mappo_mpe_spread_fixed500_1000k_eval250_20261020_02`
was stopped by user request: last confirmed optimized/checkpoint step95000,
last sampled batch95250. Logs/config/source/checkpoints are retained; this is
an incomplete run, not a1Mresult or an exact-resume guarantee. No new run starts
automatically. The base MPE formal_authorized_methods remains empty; the
completed MAPPO40 and Pure100 runs used dedicated authorization configs.

MPE MAPPO / Pure VLM / Soft-Hard VLM4Coop support is implemented for three silent
agents and three landmarks, with velocity vectors and speed labels in the
teacher RGB. See [MPE protocol and usage](docs/MPE.md).
The current user-specified renderer uses thick fixed half-radius arrows inside
circles (A0 red, A1 orange, A2 blue), internal A/V labels and green circular G
targets with unoutlined names, without external text/margins.
[Palette preview with A2 moving downward (v4)](results/mpe_rgb_down_preview_20261020_01/mpe_velocity_rgb.png).
Current v7 uses fixed1440×1440 images enclosing the full scene. Layers are target circles → target names → agent circles → agent
arrows/names/speed labels; ascending IDs within each layer. Agents can hide targets.
Ten targeted Docker checks passed before the subsequently authorized runs.
Speed labels now retain two decimal places; five focused checks passed.
[Current fixed1440×1440 real preview, V=1.53](results/mpe_square1440_preview_20261020_01/mpe_velocity_rgb.png).
The first MPE MAPPO formal 1M run completed and passed raw-log validation in
`results/mappo_mpe_spread_fixed500_1000k_20261020_01` (10 environments, 25 steps,
500 train seeds and 100 fixed evaluation seeds). Final native team return is
-54.67 ± 13.39 (100 episodes, sample SD); final all-three coverage is 0/100.
This historical formal run used10workers; a later small teacher API probe also
completed. The later Pure100 formal result is listed above. Initial MPE tests:114 passed in
135.17 s, `reports/mpe_20261020/pytest_final.log`; the 376-array LBF witness is
still exactly equal. Launch verification and frozen artifacts are in the run.
The renderer-only v2 correction additionally passed seven targeted checks;
[current real velocity screenshot](results/mpe_velocity_preview_20261020_03/mpe_velocity_rgb.png).
The completed MAPPO keeps its prelaunch source snapshot and did not use RGB.

The behavior-preserving multi-environment refactor is complete. Core owners are
`envs/`, `teachers/`, `algorithms/`, `training/`, `evaluation/`, and `utils/`;
CLI implementations are under `cli/`. Retired compatibility entries are backed
up under `src/old/vlm4coop/` and `scripts/old/`; obsolete source-patching scripts
are in `scripts/old/legacy/`. New runs and reproductions use canonical module
paths or categorized script paths, as listed in [the entry-point guide](docs/ENTRYPOINTS.md). See [the architecture guide](docs/ARCHITECTURE.md)
and [validation report](reports/refactor_20261020/REPORT.md).
The earlier refactor/entry archive preserved all LBF configuration bytes and
passed 99 tests. MPE now uses a separate YAML and isolated dependency layer;
existing LBF results and runtime are unchanged.

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
  UTC+8 and completed 1,000,062 steps, sharing the existing cache:
  `results/hard_8x8_fixed500_1000k_api_20261019_02`.
  Its `_analysis/validation.json` reports passed; no Hard container is running.
  One launch check passed without reading intermediate scores. This was a fresh
  initialization, not exact checkpoint continuation. See
  `reports/HARD_ACCEPT_REASONING_RUN.md`.
- Earlier entry-archive tests: 99 passed in CPU Docker, 73.56 s:
  `reports/archive_entries_20261020/pytest.log`. Synthetic teacher tests are not
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

## Frozen LBF student and environment protocol

All formal LBF environments are 8×8, with 3/4/5 agents and the same number of foods.
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
