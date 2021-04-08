from copy import deepcopy


class Action:

    def __init__(self, idx, action_list_size):
        self.idx = idx
        self.action_list_size = action_list_size
        self.create_action_list()
        self.action = self.action_list[idx]

    def create_action_list(self):
        """
        Swap list
        :return:
        """
        self.action_list = []
        for i in range(self.action_list_size):
            for j in range(i + 1, self.action_list_size):
                self.action_list += [(i, j)]

    def perform(self, state):
        next_state = deepcopy(state)
        next_state.pad_temporal_mapping()
        next_state.tm_swap(*self.action)  # do an action
        next_state.unpad_temporal_mapping()
        return next_state
