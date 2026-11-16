import gym_sokoban  # noqa: F401  (registers Colorbox-v1)
import gymnasium as gym
import numpy as np

from gymnasium.wrappers import ResizeObservation
from deep_sprl.experiments.abstract_experiment import AbstractExperiment


class ColoredSokobanExperiment(AbstractExperiment):
    '''
    Colored Sokoban (gym id 'Colorbox-v1' from the gym_sokoban fork): push the single box onto
    the target (t) of a 5x5 room. At every step the hidden target color is resampled (blue with
    probability CTHRES/100, yellow otherwise). Finishing yields +10 if the target color was yellow
    and -10 if it was blue, plus -0.1 per step. Without intervention the box color equals the
    target color, so the agent can wait for a yellow box before finishing the push.

    Context = [box init pos, agent init pos, intervention], each entry floored to an integer.
    The box can be in one of three locations (w-wall, t-target, e-empty)
    [[w, w, w, w, w],
     [w, e, 0, t, w],
     [w, e, 1, 2, w],
     [w, e, e, e, w],
     [w, w, w, w, w]]
    and the agent in one of five locations
    [[w, w, w, w, w],
     [w, 0, e, t, w],
     [w, 1, e, e, w],
     [w, 2, 3, 4, w],
     [w, w, w, w, w]]
    Interventions on the box color:
    0 - no intervention (box color = target color)
    1 - do(box color = blue)
    2 - do(box color = yellow)
    Interventions 1 and 2 make the box color uninformative about the reward, i.e. the optimal
    policy of such a source task is misaligned with the target task. The causally aligned bounds
    therefore exclude them. Target task: box at 1, agent at 4, no intervention.
    '''
    ENV_ID = 'Colorbox-v1'

    # target context dist is a uniform one
    TARGET_LOWER_CONTEXT_BOUNDS = np.array([1., 4., 0.])
    TARGET_UPPER_CONTEXT_BOUNDS = np.array([2., 5., 1.])

    # Original context bounds
    LOWER_CONTEXT_BOUNDS = np.array([0., 0., 0.])
    # not integers so that floor() maps the values to the correct integers
    UPPER_CONTEXT_BOUNDS = np.array([2.999, 4.999, 2.999])

    # Causally aligned context bounds
    C_LOWER_CONTEXT_BOUNDS = np.array([0., 0., 0.])
    C_UPPER_CONTEXT_BOUNDS = np.array([2.999, 4.999, 0.999])

    # Probability (in %) of a blue target color
    CTHRES = 75

    # CurrOT parameters
    # ~50% maximum rewards
    DELTA = 5.0
    # ~5% maximum l2 distance in context space
    METRIC_EPS = 1.25
    # num of episodes before update teacher distribution
    EP_PER_UPDATE = 50
    NUM_INIT_SAMPLES = 50

    # PLR params
    PLR_BUFFER_SIZE = 100
    PLR_RHO = 0.45
    PLR_BETA = 0.15
    PLR_REPLAY_RATE = 0.85

    # ALP_GMM params
    # {.1, .2, .3} x {50, 100, 200} x {500, 1000, 2000}
    AG_P_RAND = 0.3
    AG_FIT_RATE = 100
    AG_MAX_SIZE = 500

    # Goal_GAN params
    # {0.025, 0.05, 0.1} x {50, 100, 200} x {0.1, 0.2, 0.3}
    GG_NOISE_LEVEL = 0.025
    GG_FIT_RATE = 100
    GG_P_OLD = 0.3

    def create_base_env(self):
        env = gym.make(self.ENV_ID,
                       color_threshold=self.CTHRES,
                       render_mode='rgb_array',
                       disable_env_checker=True,
                       reward_box_on_target=0,
                       reward_finished=10,
                       )
        return ResizeObservation(env, 84)  # (80, 80, 3) RGB -> (84, 84, 3)

    def get_env_name(self):
        return self.ENV_ID + '-' + str(self.CTHRES)

    @staticmethod
    def count_misaligned(contexts):
        # context_2 is the intervention on the box color
        return np.sum(np.array(contexts['context_2']) >= 1)
