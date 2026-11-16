#!/usr/bin/env bash
# Sequentially run all curricula for one environment and seed: the default curriculum (train on
# the target task directly; run once, as it only samples target tasks), then ALP-GMM, GoalGAN, PLR
# and CurrOT, each without and with --causal. Each run trains for 300K steps with 16 parallel envs and
# 30 steps per env per update (625 PPO updates). Extra run.py args come last and override these
# settings (e.g. --num-envs 64).
#
# Usage: scripts/run_experiments.sh <colored_sokoban|button_maze> [seed (default 6)] [extra run.py args...]
# Example: scripts/run_experiments.sh button_maze 6 --device cuda --device-id 0
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <colored_sokoban|button_maze> [seed (default 6)] [extra run.py args...]" >&2
    exit 1
fi

ENV_NAME="$1"
shift
SEED=6
if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    SEED="$1"
    shift
fi

case "${ENV_NAME}" in
    colored_sokoban) LR=2.5e-4 ;;
    button_maze) LR=1.5e-4 ;;
    *)
        echo "Unknown environment '${ENV_NAME}' (expected colored_sokoban or button_maze)" >&2
        exit 1
        ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
COMMON_ARGS=(--env "${ENV_NAME}" --seed "${SEED}" --total-timesteps 300000 --learning-rate "${LR}"
             --num-envs 16 --num-steps 30)

"${PYTHON}" -u "${ROOT_DIR}/run.py" --type default "${COMMON_ARGS[@]}" "$@"
for TYPE in alp_gmm goal_gan plr currot; do
    "${PYTHON}" -u "${ROOT_DIR}/run.py" --type "${TYPE}" "${COMMON_ARGS[@]}" "$@"
    "${PYTHON}" -u "${ROOT_DIR}/run.py" --type "${TYPE}" --causal "${COMMON_ARGS[@]}" "$@"
done
