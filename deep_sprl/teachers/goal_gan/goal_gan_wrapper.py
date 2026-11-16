from deep_sprl.environments.base_wrapper import BaseWrapper


class GoalGANWrapper(BaseWrapper):

    def done_callback(self, step, context, discounted_reward):
        # GoalGAN is trained on binary success labels reported by the env in info['success']
        self.teacher.update(context, float(step[4]["success"]))
