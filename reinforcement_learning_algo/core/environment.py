import gym
import numpy as np
from gym import spaces, logger
from gym.utils import seeding

from reinforcement_learning_algo.core.action import Action
from reinforcement_learning_algo.cost_esimator import *
from reinforcement_learning_algo.core.state import TemporalMappingState


class Environment(gym.Env):
    """
      Description:
          A pole is attached by an un-actuated joint to a cart, which moves along
          a frictionless track. The pendulum starts upright, and the goal is to
          prevent it from falling over by increasing and reducing the cart's
          velocity.
      Observation:
          Type: Box(2)
          Num     Observation               Min                     Max
          1       Utilization               0                       1
          2       Energy                    0                       infinity
      Actions:
          Type: Discrete(N!)
          Note: Each action is transposition of the two pair of values in temporal mapping ordering,
          the number of it depends on the temporal mapping size. For now, we decided o use N*(N+1)/2
      Reward:
          Reward is an utilization value
      Starting State:
          Temporal mapping is a shuffled randomly.
          Utilization and energy values are in range [0, 1] (later, they are changed to 0 to inf.
      Episode Termination (example) :
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
      Solved Requirements (example):
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
      """

    def __init__(self, layer, im2col_layer=None, layer_rounded=None,
                 spatial_loop_comb=None, input_settings=None, mem_scheme=None, ii_su=None, mac_costs=None,
                 observation_state_length=22, utilization_threshold = 0.8, timestamp_threshold = 50):
        # Spaces
        self.observation_state_length = observation_state_length
        self.observation_space = spaces.Dict({
            'temporal_mapping': spaces.Tuple((
                spaces.Discrete(8),  # number of loops
                spaces.Discrete(np.finfo(np.float32).max)  # max primary factor value
            )),
            'energy': spaces.Discrete(1),
            'utilization': spaces.Discrete(1),
        })
        self.action_state_length = int((observation_state_length * (observation_state_length - 1)) / 2)
        self.action_space = spaces.Discrete(self.action_state_length)

        # Evaluate function optimizer variables
        self.layer = layer
        self.layer_architecture = layer.size_list_output_print
        self.im2col_layer = im2col_layer
        self.layer_rounded = layer_rounded
        self.spatial_loop_comb = spatial_loop_comb
        self.input_settings = input_settings
        self.mem_scheme = mem_scheme
        self.ii_su = ii_su
        self.mac_costs = mac_costs

        # Thresholds
        self.utilization_threshold = utilization_threshold
        self.timestamp_threshold = timestamp_threshold

        # Process variables
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, timestemp=None):
        """
        Description:
            Run one timestep of the environment's dynamics. When end of episode is reached, you are responsible for
            calling `reset()` to reset this environment's state.
            Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        # print("self.action_state_length", self.action_state_length)
        self.action = Action(action, action_list_size=self.observation_state_length)

        temporal_mapping_obj = self.state["temporal_mapping"]
        temporal_mapping_obj = self.action.perform(temporal_mapping_obj)

        energy, utilization = get_temporal_loop_estimation(temporal_mapping_obj.value, self.input_settings,
                                                           self.spatial_loop_comb,
                                                           self.mem_scheme,
                                                           layer=[self.im2col_layer, self.layer_rounded],
                                                           mac_costs=self.mac_costs)
        self.state = temporal_mapping_obj.get_state_dict(energy=energy, utilization=utilization)

        done = bool(
            utilization > self.utilization_threshold or timestemp > self.timestamp_threshold
        )

        if not done:
            reward = utilization
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = utilization
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return self.state, reward, done, {}

    def reset(self):
        """
        Description:
            Resets the environment to an initial state and returns an initial observation.
        Note:
             that this function should not reset the environment's random number generator(s); random variables in
             the environment's state should be sampled independently between multiple calls to `reset()`.
             In other words, each call of `reset()` should yield an environment suitable for a new episode,
             independent of previous episodes.
        Returns:
            observation (object): the initial observation.
        """
        temporal_mapping_obj = TemporalMappingState(layer_architecture=self.layer_architecture)
        temporal_mapping = temporal_mapping_obj.value
        energy, utilization = get_temporal_loop_estimation(temporal_mapping, self.input_settings,
                                                           self.spatial_loop_comb,
                                                           self.mem_scheme,
                                                           layer=[self.im2col_layer, self.layer_rounded],
                                                           mac_costs=self.mac_costs)
        self.state = temporal_mapping_obj.get_state_dict(energy=energy, utilization=utilization)
        # print("env_state", self.state, type(self.state))
        return self.state

    def render(self, mode='human'):
        raise NotImplementedError
