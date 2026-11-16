"""
Train a PPO agent with a curriculum on Colored Sokoban or Button Maze, or evaluate saved checkpoints.

    python run.py --env colored_sokoban --type currot --causal --seed 6
    python run.py --eval-dir logs/<timestamp>-logs/<Colorbox-v1-75|ButtonMaze-v1-20>/[causal_]<curriculum>[_allow-target]/ppo[_<params>]/seed-6/checkpoints

Unknown arguments of the form `--KEY VALUE` override experiment hyperparameters
(see AbstractExperiment.process_parameters), e.g. `--DELTA 4.0`.
"""
import os
import sys
import signal
import json
import torch
import random
import argparse
import numpy as np

from datetime import datetime
from deep_sprl.experiments import ColoredSokobanExperiment, ButtonMazeExperiment
from deep_sprl.util.parameter_parser import parse_parameters

EXPERIMENTS = {
    "colored_sokoban": ColoredSokobanExperiment,
    "button_maze": ButtonMazeExperiment,
}


def main():
    parser = argparse.ArgumentParser(description="Causally Aligned Curriculum Learning experiments")
    # General training arguments
    parser.add_argument(
        "--base-log-dir",
        type=str,
        default="./logs",
        help='path to base log dir'
    )
    parser.add_argument(
        "--type",
        type=str,
        default="default",
        choices=["default", "alp_gmm", "goal_gan", "plr", "currot"],
        help="type of curriculum learning algo to use. default - no curriculum"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="colored_sokoban",
        choices=list(EXPERIMENTS.keys()),
        help='which env to run'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=6,
        help='random seed'
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help='devices on which the experiment runs'
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="0",
        help='GPU id that the experiment runs on, e.g., "0", "0,1,3"'
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="the learning rate of the optimizer"
    )
    parser.add_argument(
        "--target-learning-rate",
        type=float,
        default=5e-5,
        help="the final learning rate if learning rate annealing is used"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=25000000,
        help="total timesteps of the experiments"
    )
    parser.add_argument(
        "--torch-deterministic",
        action='store_true',
        default=False,
        help="if toggled, `torch.use_deterministic_algorithms(True)` (and CUBLAS_WORKSPACE_CONFIG=:4096:8 if unset); "
             "with --eval-dir, also used if the run was trained with it"
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="if toggled, train for a single update (followed by one evaluation) and print debug info; "
             "logs go to <base-log-dir>/<timestamp>-debug-logs"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=512,
        help="the number of parallel game environments (used for both training and evaluation)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=64,
        help="the number of steps to run in each environment per policy rollout"
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=8,
        help="the number of mini-batches"
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=4,
        help="the K epochs to update the policy"
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        default=False,
        help="if toggled, restrict the curriculum to the causally aligned context bounds"
    )
    parser.add_argument(
        "--allow-target-task",
        action="store_true",
        default=False,
        help="if toggled, alp_gmm/goal_gan/plr/currot may also propose target-task contexts (inside "
             "TARGET_*_CONTEXT_BOUNDS) for training; by default these are rejected and resampled. "
             "No effect on --type default or on evaluation"
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="if given, only evaluate the checkpoints in this directory (<run>/checkpoints)"
    )
    parser.add_argument(
        "--eval-last",
        type=int,
        default=15,
        help="the last (eval-last) checkpoints are evaluated."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="evaluate and save a checkpoint every this many updates"
    )

    # Algorithm specific arguments
    parser.add_argument(
        "--anneal-lr",
        action='store_true',
        default=False,
        help="If toggled, learning rate annealing for policy and value networks will be used"
    )
    parser.add_argument(
        "--gae",
        action='store_false',
        default=True,
        help="If toggled, GAE for advantage computation will not be used"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.995,
        help="the discount factor gamma in reward calculation"
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda for the general advantage estimation"
    )
    parser.add_argument(
        "--norm-adv",
        action='store_false',
        default=True,
        help="If toggled, advantages normalization will not be used"
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient"
    )
    parser.add_argument(
        "--clip-vloss",
        action='store_false',
        default=True,
        help="If toggled, clipped loss for the value function will not be used."
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.02,
        help="coefficient of the entropy"
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.25,
        help="coefficient of the value function"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="the maximum norm for the gradient clipping"
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.01,
        help="the target KL divergence threshold"
    )

    args, remainder = parser.parse_known_args()
    if args.debug:
        print(args)
        print(remainder)
    parameters = parse_parameters(remainder)

    # Post process args
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    # Set visible GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)

    if args.eval_dir is None:
        train(args, parameters)
    else:
        evaluate(args, parameters)


def train(args, parameters):
    # Add time stamp to the log dir to avoid log overwritten by multiple runs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M-%S")
    if args.debug:
        args.base_log_dir = os.path.join(args.base_log_dir, f"{timestamp}-debug-logs")
    else:
        args.base_log_dir = os.path.join(args.base_log_dir, f"{timestamp}-logs")

    print('Start to train.')
    print(f'Curriculum Algo: {args.type} | Env: {args.env} | Causal: {args.causal} | '
          f'Allow target: {args.allow_target_task} | Device: {args.device} | Deterministic: {args.torch_deterministic}')
    # Built before creating the log dir so that invalid overrides do not leave an empty directory behind
    exp = EXPERIMENTS[args.env](args, parameters)
    os.makedirs(args.base_log_dir, exist_ok=True)

    set_deterministic(args.torch_deterministic)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    try:
        exp.train()
    except KeyboardInterrupt as e:
        clean_up()
        raise e
    print('Finished training!')

    # Save the contexts sampled during training (one list per sub-environment) for post analysis
    with open(os.path.join(args.base_log_dir, f'clist-{args.env}.json'), "w") as fp:
        json.dump(exp.train_contexts, fp)


def evaluate(args, parameters):
    """Evaluate the last `args.eval_last` checkpoints of a run on the target tasks and report the best."""
    eval_dir = os.path.normpath(os.path.expanduser(args.eval_dir))
    pts = [pt for pt in os.listdir(eval_dir) if pt.endswith('.pt')]
    pts = sorted(pts, key=comparison_helper, reverse=True)[:args.eval_last]

    # The training arguments are stored next to the checkpoints directory
    with open(os.path.join(os.path.dirname(eval_dir), 'args.json'), 'r') as f:
        original_args = json.load(f)
    args.env = original_args['env']
    args.causal = original_args['causal']
    args.allow_target_task = original_args.get('allow_target_task', False)
    args.type = original_args['type']
    args.torch_deterministic = args.torch_deterministic or original_args['torch_deterministic']
    exp = EXPERIMENTS[args.env](args, parameters)

    print(f'Start to evaluate {eval_dir}')
    print(f'Curriculum Algo: {args.type} | Env: {args.env} | Causal: {args.causal} | '
          f'Allow target: {args.allow_target_task} | Device: {args.device} | Deterministic: {args.torch_deterministic}')
    set_deterministic(args.torch_deterministic)
    best_score = -np.inf
    best_pt = ''
    for pt in pts:
        print(f'Evaluating {pt}...')
        # Same evaluation randomness (target contexts, Sokoban box colors, actions) for every checkpoint, set by --seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        score = exp.evaluate(os.path.join(eval_dir, pt), mode='sample')
        if score > best_score:
            best_score = score
            best_pt = pt
    print(f'Best score: {best_score} \nBest pt: {os.path.join(eval_dir, best_pt)}')
    print('====================================')


def set_deterministic(enabled):
    """`torch.use_deterministic_algorithms(enabled)`. Older PyTorch versions raise an error in deterministic
    mode on CUDA >= 10.2 unless CUBLAS_WORKSPACE_CONFIG is set, so set it if the user has not."""
    if enabled:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(enabled)


def clean_up():
    import gc
    torch.cuda.empty_cache()
    gc.collect()


def signal_handler(signum, frame):
    clean_up()
    sys.exit(0)


def comparison_helper(pt):
    # checkpoints are named '<update>-<global step>.pt' and 'final.pt'
    if 'final' in pt:
        return np.inf
    else:
        return int(pt.split('-')[0])


if __name__ == "__main__":
    clean_up()
    signal.signal(signal.SIGTERM, signal_handler)
    main()
