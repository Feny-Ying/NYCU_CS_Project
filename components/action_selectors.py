import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!
        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions
    # def get_action_distribution(self, agent_inputs, avail_actions, t_env, test_mode=False):
    #     self.epsilon = self.schedule.eval(t_env)
    #     if test_mode:
    #         self.epsilon = self.args.evaluation_epsilon
    #
    #     masked_q_values = agent_inputs.clone()
    #     masked_q_values[avail_actions == 0.0] = -float("inf")
    #
    #     # Calculate probabilities
    #     num_actions = agent_inputs.size(2)
    #     greedy_actions = masked_q_values.max(dim=2)[1]
    #     probabilities = th.full_like(agent_inputs, fill_value=self.epsilon / num_actions)
    #
    #     # Set greedy action probability
    #     batch_range = th.arange(agent_inputs.size(0)).unsqueeze(1).repeat(1, agent_inputs.size(1))
    #     agent_range = th.arange(agent_inputs.size(1)).unsqueeze(0).repeat(agent_inputs.size(0), 1)
    #     probabilities[batch_range, agent_range, greedy_actions] += (1 - self.epsilon)
    #     print(f'EpsilonGreedyActionSelector.get_action_distribution: probabilities: {probabilities}')
    #     return probabilities

REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class SoftPoliciesSelector():

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        m = Categorical(agent_inputs)
        picked_actions = m.sample().long()
        return picked_actions


REGISTRY["soft_policies"] = SoftPoliciesSelector