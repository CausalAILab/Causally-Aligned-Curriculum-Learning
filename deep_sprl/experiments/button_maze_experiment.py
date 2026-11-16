import gymnasium as gym
import numpy as np

from gymnasium.wrappers import ResizeObservation
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from deep_sprl.experiments.abstract_experiment import AbstractExperiment


class ButtonMazeExperiment(AbstractExperiment):
    '''
    Button Maze (`deep_sprl.environments.button_maze.ButtonMazeEnv`, gym id 'ButtonMaze-v1') on a
    6x6 grid with at most 30 steps. Observations are the fully observable MiniGrid encoding
    resized to 84x84x3.

    Context = [5 wall bits, fix button, fix light, agent start cell] (8 dims, floored to integers;
    fix light is 1 iff its raw value is > 0.25). With the bounds below all walls are present and
    the button is never fixed, so the curriculum varies the fix light entry and the agent start
    cell. Fixing the light, do(light = green), makes the
    displayed light uninformative about the true one, i.e. the optimal policy of such a source
    task is misaligned with the target task. The causally aligned bounds therefore keep the fix
    light entry in [0, 0.25]. Target task: all walls, button and light not fixed, agent at cell 0.
    '''
    ENV_ID = 'ButtonMaze-v1'

    # target context dist is a uniform one
    TARGET_LOWER_CONTEXT_BOUNDS = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    TARGET_UPPER_CONTEXT_BOUNDS = TARGET_LOWER_CONTEXT_BOUNDS + 0.999
    TARGET_UPPER_CONTEXT_BOUNDS[-2] = .25

    # Original context bounds
    LOWER_CONTEXT_BOUNDS = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    # not integers so that floor() maps the values to the correct integers
    # the agent can start from anywhere
    UPPER_CONTEXT_BOUNDS = np.array([1.999, 1.999, 1.999, 1.999, 1.999, 0.999, 1.999, 10.999])

    # Causally aligned context bounds: only the light cannot be fixed
    C_LOWER_CONTEXT_BOUNDS = LOWER_CONTEXT_BOUNDS.copy()
    C_UPPER_CONTEXT_BOUNDS = UPPER_CONTEXT_BOUNDS.copy()
    C_UPPER_CONTEXT_BOUNDS[-2] = .25

    # RATIO/100 chance of green light (before the button is pressed)
    RATIO = 20

    # CurrOT parameters
    # ~50% maximum rewards
    DELTA = .65
    # ~5% maximum l2 distance in context space among all axis
    # this is big since the agent position can be up to 10
    METRIC_EPS = 0.05 * (UPPER_CONTEXT_BOUNDS[-1])**2
    # num of episodes before update teacher distribution
    EP_PER_UPDATE = 50
    NUM_INIT_SAMPLES = 100

    # PLR params
    PLR_BUFFER_SIZE = 1000
    PLR_RHO = 0.15  # staleness
    PLR_BETA = 0.15  # temperature
    PLR_REPLAY_RATE = 0.85

    # ALP_GMM params
    # {.1, .2, .3} x {50, 100, 200} x {500, 1000, 2000}
    AG_P_RAND = 0.2
    AG_FIT_RATE = 50
    AG_MAX_SIZE = 1000

    # Goal_GAN params
    # {0.025, 0.05, 0.1, 0.2} x {50, 100, 200} x {0.1, 0.2, 0.3}
    GG_NOISE_LEVEL = 0.05
    GG_FIT_RATE = 75  # how many context samples per update
    GG_P_OLD = 0.2  # the bigger, the more samples from buffer

    def create_base_env(self):
        env = gym.make(self.ENV_ID,
                       ratio=self.RATIO,
                       render_mode='rgb_array',
                       max_steps=30,
                       disable_env_checker=True,
                       size=6,
                       )
        env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        return ResizeObservation(env, 84)  # (6, 6, 3) grid encoding -> (84, 84, 3)

    def get_env_name(self):
        return self.ENV_ID + '-' + str(self.RATIO)

    @staticmethod
    def count_misaligned(contexts):
        # context_1 is the fix light entry
        return np.sum(np.array(contexts['context_1']) > 0.25)
