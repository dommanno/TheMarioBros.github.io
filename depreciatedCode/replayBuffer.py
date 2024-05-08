# Replay Buffer class for storing and retrieving sampled experiences
class ReplayBuffer:
    # Constructor for Replay Buffer class
    def __init__(self, env):
        # Initialising memory count and creating arrays to store experiences
        self.mem_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)

    # Function to add experiences to the memory buffer
    def add(self, state, action, reward, state_, done):
        # If the memory count is at its max size, overwrite previous values
        if self.mem_count < MEM_SIZE:
            mem_index = self.mem_count  # Using mem_count if less than max memory size
        else:
            # Avoiding catastrophic forgetting - retain initial 10% of the replay buffer
            mem_index = int(self.mem_count % ((1-MEM_RETAIN) * MEM_SIZE) + (MEM_RETAIN * MEM_SIZE))

        # Adding the states to the replay buffer memory
        self.states[mem_index]  = state     # Storing the state
        self.actions[mem_index] = action    # Storing the action
        self.rewards[mem_index] = reward    # Storing the reward
        self.states_[mem_index] = state_    # Storing the next state
        self.dones[mem_index] =  1 - done   # Storing the done flag
        self.mem_count += 1  # Incrementing memory count

    # Function to sample random batch of experiences
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True).to(self.device)

        states = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones = self.dones[batch_indices]

        # Returning the random sampled experiences
        return np.array(states), np.array(actions), np.array(rewards), np.array(states_), np.array(dones)
    
    def __len__(self):
        return self.mem_count