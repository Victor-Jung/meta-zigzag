from copy import deepcopy

import gym
import numpy as np
from gym import spaces, logger
from gym.utils import seeding
import gifmaker

from reinforcement_learning_algo.core.state import TemporalMappingState
from reinforcement_learning_algo.cost_esimator import *
from reinforcement_learning_algo.utils.renderer import make_frame, make_layout, make_figure_from_frames, add_sliders


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
          0       Temporal mapping
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
                 spatial_loop_comb=None, input_settings=None, mem_scheme=None, ii_su=None, spatial_unrolling=None, mac_costs=None,
                 observation_state_length=22, utilization_threshold=0.8, timestamp_threshold=50,
                 repetition_threshold=5):
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

        # ZigZg evaluate function optimizer variables
        self.layer = layer
        self.layer_architecture = layer.size_list_output_print
        self.im2col_layer = im2col_layer
        self.layer_rounded = layer_rounded
        self.spatial_loop_comb = spatial_loop_comb
        self.input_settings = input_settings
        self.mem_scheme = mem_scheme
        self.ii_su = ii_su
        self.mac_costs = mac_costs
        self.spatial_unrolling = spatial_unrolling

        # Thresholds
        self.utilization_threshold = utilization_threshold
        self.timestamp_threshold = timestamp_threshold
        self.repetition_threshold = repetition_threshold

        # Process variables
        self.state = None
        self.steps_beyond_done = None

        self.last_actions = []
        self.last_reward = 0

        # Render part
        self.frames = []
        self.layout = make_layout()
        self.steps = 0

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

    def calculate_reward(self, energy, utilization, timestemp):
        reward = utilization
        # print(reward, self.last_reward)
        # reward = 1/ pow(reward - self.last_reward, 2) if reward - self.last_reward > 0 else 0
        # reward = reward - self.last_reward
        # self.last_reward = reward
        # reward = 1 if reward > self.last_reward else 0
        # self.last_reward = utilization

        return reward

    def calculate_costs(self, temporal_mapping):
        return get_temporal_loop_estimation(temporal_mapping, self.input_settings,
                                            self.spatial_loop_comb,
                                            self.mem_scheme,
                                            layer=[self.im2col_layer, self.layer_rounded],
                                            mac_costs=self.mac_costs)

    def step(self, action, timestep=None, episode=None):
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
        # print(action)
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action.idx), err_msg

        self.action = action
        self.timestep = timestep

        is_repetition = False

        self.last_actions.append(action.idx)
        if len(self.last_actions) > self.repetition_threshold:
            self.last_actions.pop(0)
        if len(self.last_actions) >= self.repetition_threshold and len(set(self.last_actions)) < 3:
            is_repetition = True

        temporal_mapping_obj = self.state["temporal_mapping"]
        temporal_mapping_obj = self.action.perform(temporal_mapping_obj)

        energy, utilization = self.calculate_costs(temporal_mapping_obj.value)
        self.state = temporal_mapping_obj.get_state_dict(energy=energy, utilization=utilization)
        self.utilization = utilization

        done = bool(
            utilization > self.utilization_threshold
            or timestep > self.timestamp_threshold or is_repetition
        )

        if not done:
            reward = self.calculate_reward(energy, utilization, timestep)
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = self.calculate_reward(energy, utilization, timestep)
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        self.steps += 1
        return self.state, reward, done, utilization

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
        self.steps_beyond_done = None
        self.previous_reward = 0
        temporal_mapping_obj = TemporalMappingState(self.spatial_unrolling, layer_architecture=self.layer_architecture)
        temporal_mapping = temporal_mapping_obj.value
        energy, utilization = get_temporal_loop_estimation(temporal_mapping, self.input_settings,
                                                           self.spatial_loop_comb,
                                                           self.mem_scheme,
                                                           layer=[self.im2col_layer, self.layer_rounded],
                                                           mac_costs=self.mac_costs)
        self.state = temporal_mapping_obj.get_state_dict(energy=energy, utilization=utilization)
        self.last_actions = []
        return self.state

    def render_frame(self):
        action = self.action
        temporal_mapping = self.state['temporal_mapping']

        utilization = self.utilization
        swap_list = self.action.generate_swap_list()
        x = [pair[0] for pair in swap_list]
        y = [pair[1] for pair in swap_list]
        z = []
        for swap in swap_list:
            temporal_mapping_obj = deepcopy(temporal_mapping)
            temporal_mapping_obj = temporal_mapping_obj.tm_swap(swap[0], swap[1])
            energy, utilization = self.calculate_costs(temporal_mapping_obj.value)
            z.append(utilization)

        point_x = action.action[0]
        point_y = action.action[1]
        point_z = utilization
        self.frame = make_frame(temporal_mapping=temporal_mapping_obj.value, frame_id=self.steps, x=x, y=y, z=z,
                                point_x=point_x, point_y=point_y, point_z=point_z)
        self.frames.append(self.frame)
        return self.frame

    def render(self, mode='human'):
        self.figure = make_figure_from_frames(self.frames)
        self.figure, layout = add_sliders(self.figure, self.layout)
        # raise NotImplementedError

    def save_animation(self):
        fp = open("out.gif", "wb")
        # gifmaker.makedelta(fp, self.frames)
        # fp.close()

    def display(self):
        self.figure.show()