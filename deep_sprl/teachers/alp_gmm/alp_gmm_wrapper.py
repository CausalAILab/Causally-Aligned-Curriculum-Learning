from deep_sprl.environments.base_wrapper import BaseWrapper


class ALPGMMWrapper(BaseWrapper):

    def done_callback(self, step, context, discounted_reward):
        self.teacher.update(context, discounted_reward)
