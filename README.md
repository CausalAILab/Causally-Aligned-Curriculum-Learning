# Causally Aligned Curriculum Learning

Code for [Causally Aligned Curriculum Learning](https://arxiv.org/abs/2503.16799)
(Mingxuan Li, Junzhe Zhang, Elias Bareinboim; ICLR 2024).

Curriculum generators build source tasks by editing parameters ("contexts") of the target task. Under
unobserved confounding, some edits change the optimal policy, and training on them hurts performance on the
target. Passing `--causal` restricts each generator (ALP-GMM, Goal-GAN, PLR, CurrOT) to causally aligned
source tasks, as defined by the `C_*_CONTEXT_BOUNDS` of each experiment. The student is PPO.

## Environments

| `--env` | Task | Context | Removed by `--causal` |
|---|---|---|---|
| `colored_sokoban` | 5x5 Sokoban with one box. A hidden confounder redraws the box color every step. Solving gives +10 if the box is yellow (25% of steps) and -10 if it is blue. | box start, agent start, box-color intervention | box-color intervention |
| `button_maze` | 6x6 MiniGrid maze. Before the button is pressed, the goal light is uninformative and the goal pays +1 with probability 0.2, otherwise -1. After the button is pressed, the light is green and the goal pays +1. | walls, button, goal-light intervention, agent start (only the last two vary) | goal-light intervention |

## Installation

```bash
# from the repository root
conda create -n causal-cl python=3.10 -y && conda activate causal-cl
pip install torch                 # choose the build for your platform/CUDA from pytorch.org
pip install -r requirements.txt
# Colored Sokoban (gymnasium port of gym-sokoban); --no-deps because its setup.py pulls legacy `gym`
# (`pip check` will therefore report the missing gym; this is expected)
pip install --no-deps "git+https://github.com/drmeerkat/gym-sokoban.git@a228b338be046b37bffa658e282c3a1b2d2f185d"
# C++/OpenMP Nadaraya-Watson regressor used by CurrOT (needs a C++ compiler with OpenMP, no CMake)
pip install ./nadaraya-watson
```

## Usage

```bash
# CurrOT on Colored Sokoban: original vs. causally aligned context space
python run.py --env colored_sokoban --type currot --seed 6 --device cuda
python run.py --env colored_sokoban --type currot --causal --seed 6 --device cuda

# All curricula for one env and seed: default, then alp_gmm/goal_gan/plr/currot with and without --causal
bash scripts/run_experiments.sh button_maze 6 --device cuda

# Evaluate the last 15 checkpoints of a run on the target task (settings are read from the run's args.json)
python run.py --eval-dir logs/<timestamp>-logs/Colorbox-v1-75/causal_currot/ppo_<params>/seed-6/checkpoints --eval-last 15
```

- `run.py` defaults to 512 parallel envs, 64 steps per update, 25M steps and lr 5e-4.
  `scripts/run_experiments.sh` uses 16 parallel envs, 30 steps per update and 300K steps, with lr 2.5e-4
  (Colored Sokoban) or 1.5e-4 (Button Maze).
- Runs are written to `./logs/<timestamp>-logs/` (`args.json`, TensorBoard events, `checkpoints/`). Change
  the location with `--base-log-dir` and view runs with `tensorboard --logdir logs`.
- `--allow-target-task` lets ALP-GMM, Goal-GAN, PLR and CurrOT also train on the target task (contexts
  inside the experiment's `TARGET_*_CONTEXT_BOUNDS`). By default such contexts are rejected and
  resampled, so these curricula train almost only on source tasks. In Button Maze, teacher samples on
  the upper edge of the target bounds are not rejected (under 2% of training episodes). The default
  curriculum and evaluation always use the target task. Runs with the option are logged under
  `[causal_]<curriculum>_allow-target`.
- Extra `--KEY VALUE` arguments override experiment hyperparameters (e.g. `--DELTA 4.0`). See
  `python run.py --help` for all options.
- `--seed` (default 6) fixes the environment, curriculum, network and evaluation randomness, so runs
  with the same seed repeat exactly on the same machine, device and software versions (including the
  number of CPU threads). On GPU, add `--torch-deterministic` to also rule out nondeterministic CUDA
  kernels. With `--eval-dir`, every checkpoint is evaluated with the same random stream, set by `--seed`.

## Code map

```
run.py                       CLI: training and checkpoint evaluation
deep_sprl/experiments/       context bounds, causal bounds (C_*_CONTEXT_BOUNDS), teacher settings per env
deep_sprl/environments/      ButtonMazeEnv; BaseWrapper feeds teacher contexts to the env
deep_sprl/teachers/          ALP-GMM, Goal-GAN, PLR, CurrOT (spl/)
deep_sprl/students/ppo.py    PPO (based on CleanRL)
nadaraya-watson/             C++ extension used by CurrOT
```

## Acknowledgements

Builds on [CurrOT](https://github.com/psclklnk/currot) (MIT), [gym-sokoban](https://github.com/mpSchrader/gym-sokoban),
[MiniGrid](https://github.com/Farama-Foundation/Minigrid) and [CleanRL](https://github.com/vwxyzjn/cleanrl).

## Citation

```bibtex
@inproceedings{li2024causally,
  title={Causally Aligned Curriculum Learning},
  author={Li, Mingxuan and Zhang, Junzhe and Bareinboim, Elias},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```
