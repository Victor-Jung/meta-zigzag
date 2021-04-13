import itertools
from copy import deepcopy


class Action:

    def __init__(self, action_list_size, idx=None):
        self.action_list_size = action_list_size
        self.create_action_list()
        if idx is not None:
            self.set_idx(idx)

    def set_idx(self, idx):
        self.idx = idx
        self.action = self.action_list[idx]

    def create_action_list(self):
        """
        Swap list
        :return:
        """
        self.action_list = list(itertools.combinations(range(self.action_list_size), 2))

    def filter_action_list(self, state=None, action_probs=None):
        loop_types = [loop_type for loop_type, weight in state]
        for i, action in enumerate(self.action_list):
            pos1, pos2 = action
            try:
                if state[pos1] == state[pos2]:
                    action_probs[i] = 0
                else:
                    k = set(loop_types[pos1:pos2 + 1])
                    if len(k) == 1:
                        action_probs[i] = 0
            except Exception:
                action_probs[i] = 0
        return action_probs

    def perform(self, state):
        next_state = deepcopy(state)
        next_state.pad_temporal_mapping()
        next_state.tm_swap(*self.action)  # do an action
        next_state.unpad_temporal_mapping()
        return next_state

    def __str__(self):
        return str(self.__dict__)
