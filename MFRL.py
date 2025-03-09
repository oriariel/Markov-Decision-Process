import numpy as np
import random

def sarsa(w, h, L, p, r, gamma=0.5, alpha=0.1, epsilon=0.01, episodes=1000):
    # Initialize Q-table with zeros
    Q = np.zeros((h, w, 4))  # 4 possible actions: up, down, left, right

    # Define actions and their corresponding movements
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    
    # Convert L to dictionary for fast lookup
    rewards = {(x, y): reward for x, y, reward in L}

    # Check if the state is terminal
    def is_terminal(state):
        return state in rewards

    # Get the reward for a state
    def get_reward(state):
        return rewards.get(state, r)

    # Get next state given current state and action
    def get_next_state(state, action):
        x, y = state
        dx, dy = action
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            return nx, ny
        return x, y  # stay in place if hitting the wall or going out of bounds

    # Choose action using epsilon-greedy strategy
    def choose_action(state):
        x, y = state
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 3)  # random action
        else:
            return np.argmax(Q[y, x])  # best action

    # Training process
    for episode in range(episodes):
        # Start at a random non-terminal state
        x, y = random.randint(0, w-1), random.randint(0, h-1)
        while is_terminal((x, y)):
            x, y = random.randint(0, w-1), random.randint(0, h-1)

        state = (x, y)
        action = choose_action(state)
        
        steps = 0  # Add a step counter to avoid infinite loops
        while not is_terminal(state) and steps < 1000:  # Limit the number of steps per episode
            if random.uniform(0, 1) < p:
                next_state = get_next_state(state, actions[action])
            else:
                slip_action = random.choice([a for i, a in enumerate(actions) if i != action])
                next_state = get_next_state(state, slip_action)

            reward = get_reward(next_state)
            next_action = choose_action(next_state)

            # Update Q-value
            Q[state[1], state[0], action] += alpha * (
                reward + gamma * Q[next_state[1], next_state[0], next_action] - Q[state[1], state[0], action])

            # Move to the next state and action
            state = next_state
            action = next_action
            steps += 1

    # Derive policy from Q-table
    policy = np.zeros((h, w), dtype=float)
    value_function = np.zeros((h, w))

    for y in range(h):
        for x in range(w):
            if not is_terminal((x, y)):
                policy[y, x] = np.argmax(Q[y, x])
                value_function[y, x] = np.max(Q[y, x])
            else:
                policy[y, x] = -1  # Terminal states
                value_function[y, x] = 0  # Set the value for reward and penalty states to zero

    return Q, policy, value_function

# Example usage
w = 12
h = 4
L = [(1,0,-100),(2,0,-100),(3,0,-100),(4,0,-100),(5,0,-100),(6,0,-100),(7,0,-100),(8,0,-100),(9,0,-100),(10,0,-100),(11,0,1)]
p = 1
r = -1

Q, policy, value_function = sarsa(w, h, L, p, r)
print("Value Function:")
V_fliped = np.flipud(value_function)
print(V_fliped)


