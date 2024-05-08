# Replay Buffer class for storing and retrieving sampled experiences
class ReplayBuffer:
    def __init__(self, env, mem_size=MEM_SIZE):
        # Initialising memory count and creating arrays to store experiences
        self.memory = deque(maxlen=mem_size)
        self.mem_count = 0

    def add(self, state, action, reward, state_, done):
        # Adding experience to memory
        self.memory.append((state, action, reward, state_, done))
        self.mem_count += 1

    def sample(self):
        # Randomly sample a batch of experiences
        batch_size = min(BATCH_SIZE, self.mem_count)
        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, states_, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(states_), np.array(dones)
    
    def __len__(self):
        return self.mem_count