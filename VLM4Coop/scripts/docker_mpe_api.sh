#!/usr/bin/env bash
# Future explicit MPE teacher runs: key via stdin only; never Docker args/env files.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/docker_mpe.sh" --stdin \
  bash -c 'IFS= read -r OPENCODE_API_KEY; export OPENCODE_API_KEY; exec "$@"' bash "$@"
