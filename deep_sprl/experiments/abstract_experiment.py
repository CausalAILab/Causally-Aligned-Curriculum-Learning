import os
import copy
import json
import numpy as np
import gymnasium as gym

from enum import Enum
from abc import ABC, abstractmethod
from torch import optim

from deep_sprl.environments.base_wrapper import BaseWrapper
from deep_sprl.students.ppo import PPOInterface
from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
from deep_sprl.teachers.dummy_teachers import UniformSampler, DistributionSampler
from deep_sprl.teachers.goal_gan import GoalGAN, GoalGANWrapper
from deep_sprl.teachers.plr import PLR, PLRWrapper
from deep_sprl.teachers.spl import SelfPacedWrapper, CurrOT


class CurriculumType(Enum):
    Default = "default"  # no curriculum: train directly on the target task distribution
    ALPGMM = "alp_gmm"
    GoalGAN = "goal_gan"
    PLR = "plr"
    CurrOT = "currot"

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(string):
        try:
            return CurriculumType(string)
        except ValueError:
            raise RuntimeError("Invalid string: '" + string + "'")


class AbstractExperiment(ABC):
    """
    Shared experiment logic: environment / teacher construction, training and evaluation.

    Subclasses define the class-level constants (context bounds, teacher hyperparameters) and
    implement `create_base_env()`, `get_env_name()` and `count_misaligned()`. Context bounds:
      TARGET_{LOWER,UPPER}_CONTEXT_BOUNDS: target task distribution (uniform); used for evaluation
                                           and by the default curriculum. Curriculum contexts inside
                                           these bounds are rejected unless --allow-target-task.
      {LOWER,UPPER}_CONTEXT_BOUNDS:        context space available to the curriculum teachers.
      C_{LOWER,UPPER}_CONTEXT_BOUNDS:      causally aligned context space; replaces the above
                                           when running with --causal.
    Hyperparameters that can be overridden from the command line (`--KEY VALUE`) are listed in
    `process_parameters`; those in APPENDIX_KEYS are appended to the log directory name.
    """

    APPENDIX_KEYS = {
        CurriculumType.CurrOT: ["DELTA", "METRIC_EPS", "EP_PER_UPDATE", "NUM_INIT_SAMPLES"],
        CurriculumType.GoalGAN: ["GG_NOISE_LEVEL", "GG_FIT_RATE", "GG_P_OLD"],
        CurriculumType.ALPGMM: ["AG_P_RAND", "AG_FIT_RATE", "AG_MAX_SIZE"],
        CurriculumType.Default: [],
        CurriculumType.PLR: ["PLR_REPLAY_RATE", "PLR_BETA", "PLR_RHO"],
    }

    def __init__(self, args, parameters):
        self.args = args
        self.base_log_dir = args.base_log_dir
        self.parameters = parameters
        self.curriculum = CurriculumType.from_string(args.type)
        self.seed = args.seed
        self.num_envs = args.num_envs
        self.process_parameters()

        if self.args.causal:
            self.UPPER_CONTEXT_BOUNDS = self.C_UPPER_CONTEXT_BOUNDS
            self.LOWER_CONTEXT_BOUNDS = self.C_LOWER_CONTEXT_BOUNDS

    @abstractmethod
    def create_base_env(self):
        """Create the raw contextual environment with 84x84x3 image observations."""
        pass

    @abstractmethod
    def get_env_name(self):
        pass

    @staticmethod
    @abstractmethod
    def count_misaligned(contexts):
        """Number of episodes in a rollout batch whose context is a misaligned source task.
        `contexts` maps 'context_0'..'context_2' to per-episode lists (see BaseWrapper.update)."""
        pass

    @staticmethod
    def parse_max_size(val):
        if val == "None":
            return None
        else:
            return int(val)

    def process_parameters(self):
        allowed_overrides = {"DELTA": float, "METRIC_EPS": float, "EP_PER_UPDATE": int,
                             "AG_P_RAND": float, "AG_FIT_RATE": int, "AG_MAX_SIZE": self.parse_max_size,
                             "GG_NOISE_LEVEL": float, "GG_FIT_RATE": int, "GG_P_OLD": float,
                             "PLR_REPLAY_RATE": float, "PLR_BETA": float, "PLR_RHO": float}
        for key in sorted(self.parameters.keys()):
            if key not in allowed_overrides:
                raise RuntimeError("Parameter '" + str(key) + "' not allowed")
            setattr(self, key, allowed_overrides[key](self.parameters[key]))
        # CurrOT's success buffer holds NUM_INIT_SAMPLES contexts and receives EP_PER_UPDATE per update
        if self.curriculum is CurriculumType.CurrOT and self.EP_PER_UPDATE > self.NUM_INIT_SAMPLES:
            raise RuntimeError("EP_PER_UPDATE (%d) must not exceed NUM_INIT_SAMPLES (%d)"
                               % (self.EP_PER_UPDATE, self.NUM_INIT_SAMPLES))

    def get_curriculum(self):
        name = ("causal_" if self.args.causal else "") + str(self.curriculum)
        if self.args.allow_target_task and self.curriculum is not CurriculumType.Default:
            name += "_allow-target"
        return name

    def get_log_dir(self):
        learner_string = "ppo"
        for key in sorted(self.APPENDIX_KEYS[self.curriculum]):
            learner_string += "_" + key + "-" + str(getattr(self, key)).replace(" ", "")

        return os.path.join(self.base_log_dir, self.get_env_name(), self.get_curriculum(),
                            learner_string, "seed-" + str(self.seed))

    def target_sampler(self, n=None, rng=None):
        if n is None:
            n = self.EP_PER_UPDATE

        if rng is None:
            rng = np.random
        return rng.uniform(
            self.TARGET_LOWER_CONTEXT_BOUNDS,
            self.TARGET_UPPER_CONTEXT_BOUNDS,
            size=(n, len(self.TARGET_LOWER_CONTEXT_BOUNDS))
        )

    def non_target_sampler(self, rng=None):
        """Uniform sample from the curriculum context space that is not a target context."""
        if rng is None:
            rng = np.random
        context = self.TARGET_LOWER_CONTEXT_BOUNDS
        b = 0.5 * (self.UPPER_CONTEXT_BOUNDS + self.LOWER_CONTEXT_BOUNDS)
        m = 0.5 * (self.UPPER_CONTEXT_BOUNDS - self.LOWER_CONTEXT_BOUNDS)
        while np.all(context < self.TARGET_UPPER_CONTEXT_BOUNDS) and \
                np.all(context >= self.TARGET_LOWER_CONTEXT_BOUNDS):
            norm_sample = rng.uniform(low=-1, high=1, size=self.LOWER_CONTEXT_BOUNDS.shape)
            context = m * norm_sample + b
        return context

    def create_currot_teacher(self):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        init_samples = np.random.uniform(
            self.LOWER_CONTEXT_BOUNDS,
            self.UPPER_CONTEXT_BOUNDS,
            size=(self.NUM_INIT_SAMPLES, len(self.LOWER_CONTEXT_BOUNDS))
        )
        return CurrOT(bounds, init_samples, self.target_sampler, self.DELTA, self.METRIC_EPS)

    def make_env(self, evaluation, seed):
        """Return a thunk creating one (sub-)environment: base env -> teacher -> wrapper."""
        def make_env_helper():
            env = self.create_base_env()
            if self.args.allow_target_task:
                # no target bounds: the wrapper uses every teacher sample
                wrapper_kwargs = {}
            else:
                wrapper_kwargs = dict(target_ctx_upper=self.TARGET_UPPER_CONTEXT_BOUNDS,
                                      target_ctx_lower=self.TARGET_LOWER_CONTEXT_BOUNDS,
                                      non_target_sampler=self.non_target_sampler)

            if evaluation:
                # Use target context bound for evaluation!
                teacher = UniformSampler(self.TARGET_LOWER_CONTEXT_BOUNDS.copy(), self.TARGET_UPPER_CONTEXT_BOUNDS.copy())
                env = BaseWrapper(env, teacher, self.args.gamma, **wrapper_kwargs)
            elif self.curriculum is CurriculumType.Default:
                teacher = DistributionSampler(self.target_sampler, self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
                env = BaseWrapper(env, teacher, self.args.gamma, **wrapper_kwargs)
            elif self.curriculum is CurriculumType.ALPGMM:
                teacher = ALPGMM(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(), seed=self.seed,
                                 fit_rate=self.AG_FIT_RATE, random_task_ratio=self.AG_P_RAND,
                                 max_size=self.AG_MAX_SIZE)
                env = ALPGMMWrapper(env, teacher, self.args.gamma, **wrapper_kwargs)
            elif self.curriculum is CurriculumType.GoalGAN:
                samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS,
                                            size=(1000, len(self.UPPER_CONTEXT_BOUNDS)))
                teacher = GoalGAN(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(),
                                  state_noise_level=self.GG_NOISE_LEVEL, success_distance_threshold=0.01,
                                  update_size=self.GG_FIT_RATE, n_rollouts=2, goid_lb=0.25, goid_ub=0.75,
                                  p_old=self.GG_P_OLD, pretrain_samples=samples, device=self.args.device)
                env = GoalGANWrapper(env, teacher, self.args.gamma, **wrapper_kwargs)
            elif self.curriculum is CurriculumType.CurrOT:
                teacher = self.create_currot_teacher()
                env = SelfPacedWrapper(env, teacher, self.args.gamma, episodes_per_update=self.EP_PER_UPDATE,
                                       **wrapper_kwargs)
            elif self.curriculum is CurriculumType.PLR:
                teacher = PLR(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, self.PLR_REPLAY_RATE,
                              self.PLR_BUFFER_SIZE, self.PLR_BETA, self.PLR_RHO)
                env = PLRWrapper(env, teacher, self.args.gamma, **wrapper_kwargs)
            else:
                raise RuntimeError("Invalid learning type")

            env = gym.wrappers.RecordEpisodeStatistics(env)
            # env.unwrapped is the BaseWrapper
            env.unwrapped.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return make_env_helper

    def create_environment(self, evaluation=False):
        # One seed per sub-environment, derived from --seed and different for training and evaluation
        seeds = np.random.SeedSequence([self.seed, int(evaluation)]).generate_state(self.num_envs)
        return gym.vector.SyncVectorEnv([self.make_env(evaluation, int(s)) for s in seeds])

    def create_learner(self, env, eval_env):
        """Create the PPO agent wrapped with an interface that supports load, train, eval, etc."""
        return PPOInterface(env, eval_env, self.args, self.count_misaligned)

    def create_experiment(self, evaluation_only=False):
        eval_vec_env = self.create_environment(evaluation=True)
        if evaluation_only:
            return self.create_learner(None, eval_vec_env), None

        vec_env = self.create_environment(evaluation=False)
        agent = self.create_learner(vec_env, eval_vec_env)
        optimizer = optim.Adam(agent.parameters(), lr=self.args.learning_rate, eps=1e-5)
        if self.curriculum is CurriculumType.PLR:
            # PLR scores levels with the learner's value estimates
            for i in range(self.num_envs):
                vec_env.envs[i].unwrapped.learner = agent
        return agent, optimizer

    def train(self):
        agent, optimizer = self.create_experiment()
        log_directory = self.get_log_dir()

        if os.path.exists(log_directory):
            print("Log directory already exists! Skipping training.")
        else:
            os.makedirs(log_directory)
            # save all args and params into log_dir
            with open(os.path.join(log_directory, "args.json"), "w") as f:
                out = copy.copy(vars(self.args))
                out.update(self.parameters)
                f.write(json.dumps(out))

            agent.train(optimizer, log_directory)
        self.train_contexts = [env.unwrapped.train_contexts for env in agent.env.envs]

    def evaluate(self, agent_path, mode='sample'):
        agent, _ = self.create_experiment(evaluation_only=True)
        agent.load(agent_path)
        eval_return, _, _ = agent.evaluate(mode=mode, ep=128)
        # Make sure these envs are closed properly!
        agent.eval_env.close()
        print(f'Return under {mode}: {eval_return}')
        return eval_return
