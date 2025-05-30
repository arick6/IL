import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.max_size = max_size

        # Store observations and actions
        self.obs = None
        self.acs = None

    def __len__(self):
        if self.obs is not None:
            return self.obs.shape[0]
        else:
            return 0

    def add_data(self, observations, actions):
        """
        Add observations and actions to the buffer.
        """
        if self.obs is None:
            # Initialize buffer with the first batch of data
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
        else:
            # Concatenate new data and ensure size does not exceed max_size
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]

    def sample(self, batch_size):
        """
        Sample a batch of data from the buffer.
        Returns: observations, actions
        """
        assert self.__len__() >= batch_size, "Not enough samples in the buffer to sample a batch"
        indices = np.random.choice(len(self), batch_size, replace=False)
        return self.obs[indices], self.acs[indices]
