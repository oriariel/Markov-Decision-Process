import numpy as np
import random

def q_learning(w, h, L, p, r, gamma=0.5, alpha=0.1, epsilon=0.01, episodes=1000):
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

    # Get next state given current state and action, considering probability p
    def get_next_state(state, action):
        x, y = state
        if random.uniform(0, 1) < p:  # Successful movement with probability p
            dx, dy = action
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                return nx, ny
        return x, y  # stay in place if hitting the wall or going out of bounds or movement fails

    # Training process
    for episode in range(episodes):
        # Start at a random non-terminal state
        x, y = random.randint(0, w-1), random.randint(0, h-1)
        while is_terminal((x, y)):
            x, y = random.randint(0, w-1), random.randint(0, h-1)

        steps = 0  # Add step counter to avoid infinite loops
        while not is_terminal((x, y)) and steps < w * h * 10:  # Safety exit after too many steps
            state = (x, y)
            
            # Choose action using epsilon-greedy strategy
            if random.uniform(0, 1) < epsilon:
                action_index = random.randint(0, 3)
            else:
                action_index = np.argmax(Q[y, x])
            
            action = actions[action_index]
            next_state = get_next_state(state, action)
            nx, ny = next_state
            
            # Determine the immediate reward
            immediate_reward = get_reward(next_state)
            
            # Update Q-value
            best_next_action = np.argmax(Q[ny, nx])
            Q[y, x, action_index] += alpha * (
                immediate_reward + gamma * Q[ny, nx, best_next_action] - Q[y, x, action_index])
            
            # Move to the next state
            x, y = next_state
            steps += 1  # Increment step counter

        # if steps >= w * h * 10:
            # print(f"Warning: Episode {episode} terminated early due to too many steps")

    # Derive policy from Q-table
    policy = np.zeros((h, w), dtype=float)
    for y in range(h):
        for x in range(w):
            if not is_terminal((x, y)):
                policy[y, x] = np.argmax(Q[y, x])
            else:
                policy[y, x] = -1  # Terminal states

    # Create value function from Q-table
    value_function = np.max(Q, axis=2)
    for y in range(h):
        for x in range(w):
            if is_terminal((x, y)):
                value_function[y, x] = 0  # Terminal states should have a value of 0

    return Q, policy, value_function

# Example usage

w = 7
h = 7
L = [(3,1,0),(3,5,0),(1,1,-4),(1,5,-6),(5,1,1),(5,5,4)]
p = 0.8
r = -0.25

Q, policy, value_function = q_learning(w, h, L, p, r)
#print("Q-Table:")
#print(Q)
#print("Policy:")
#print(policy)
V_flipped = np.flipud(value_function)
print("Value Function:")
print(V_flipped)

