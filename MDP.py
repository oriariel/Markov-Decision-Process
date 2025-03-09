import numpy as np

def value_iteration(w, h, L, p, r, gamma=0.5, epsilon=0.01):
    # Initialize value function
    V = np.zeros((h, w))
    policy = np.zeros((h, w), dtype=int)

    # Define actions and their corresponding movements
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    
    # Convert L to dictionary for fast lookup
    rewards = {(x, y): r for x, y, r in L}

    # Check if the state is terminal
    def is_terminal(state):
        return state in rewards

    # Get the reward for a state
    def get_reward(state):
        return rewards.get(state, 0)

    # Get next state given current state and action
    def get_next_state(state, action):
        x, y = state
        dx, dy = action
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and not is_terminal((nx, ny)):
            return nx, ny
        return x, y  # stay in place if hitting the wall or going out of bounds

    # Value Iteration
    while True:
        delta = 0
        for y in range(h):
            for x in range(w):
                state = (x, y)
                if is_terminal(state):
                    continue
                v = V[y, x]
                new_v = float('-inf')
                for i, action in enumerate(actions):
                    expected_value = 0
                    for j, next_action in enumerate(actions):
                        next_state = get_next_state(state, next_action)
                        if i == j:
                            prob = p
                        else:
                            prob = (1 - p) / 2  # Adjusting the probability distribution
                        expected_value += prob * (get_reward(next_state) + gamma * V[next_state[1], next_state[0]])
                    new_v = max(new_v, expected_value + r)
                V[y, x] = new_v
                delta = max(delta, abs(v - new_v))
        if delta < epsilon:
            break

    # Extract Policy
    for y in range(h):
        for x in range(w):
            state = (x, y)
            if is_terminal(state):
                continue
            best_action = None
            best_value = float('-inf')
            for i, action in enumerate(actions):
                expected_value = 0
                for j, next_action in enumerate(actions):
                    next_state = get_next_state(state, next_action)
                    if i == j:
                        prob = p
                    else:
                        prob = (1 - p) / 2  # Adjusting the probability distribution
                    expected_value += prob * (get_reward(next_state) + gamma * V[next_state[1], next_state[0]])
                if expected_value + r > best_value:
                    best_value = expected_value + r
                    best_action = i
            policy[y, x] = best_action

    return V, policy

# Example usage
w = 4
h = 3
L = [(1, 1, 0), (3, 2, 1), (3, 1, -1)]
p = 0.8
r = -0.04

V, policy = value_iteration(w, h, L, p, r)
print("MDP = test #1:")
print("Value Function:")
V_fliped = np.flipud(V)
print(V_fliped)


# w = 4
# h = 3
# L = [(1,1,0),(3,2,1),(3,1,-1)]
# p = 0.8
# r = 0.04
# V, policy = value_iteration(w, h, L, p, r)
# print("test #2:")
# print("Value Function:")
# V_fliped = np.flipud(V)
# print(V_fliped)

# w = 4
# h = 3
# L = [(1,1,0),(3,2,1),(3,1,-1)]
# p = 0.8
# r = -1
# V, policy = value_iteration(w, h, L, p, r)
# print("test #3:")
# print("Value Function:")
# V_fliped = np.flipud(V)
# print(V_fliped)

# w = 12
# h = 4
# L = [(1,0,-100),(2,0,-100),(3,0,-100),(4,0,-100),(5,0,-100),(6,0,-100),(7,0,-100),(8,0,-100),(9,0,-100),(10,0,-100),(11,0,0)]
# p = 1
# r = -1
# V, policy = value_iteration(w, h, L, p, r)
# print("test #4:")
# print("Value Function:")
# V_fliped = np.flipud(V)
# print(V_fliped)

# w = 12
# h = 6
# L = [(1,0,-100),(2,0,-100),(3,0,-100),(4,0,-100),(5,0,-100),(6,0,-100),(7,0,-100),(8,0,-100),(9,0,-100),(10,0,-100),(11,0,0)]
# p = 0.9
# r = -1
# V, policy = value_iteration(w, h, L, p, r)
# print("test #5:")
# print("Value Function:")
# V_fliped = np.flipud(V)
# print(V_fliped)
# #t6

# w = 5
# h = 5
# L = [(4,0,-10),(0,4,-10),(1,1,1),(3,3,2)]
# p = 0.9
# r = -0.5
# V, policy = value_iteration(w, h, L, p, r)
# print("test #6:")
# print("Value Function:")
# V_fliped = np.flipud(V)
# print(V_fliped)
#t7

# w = 5
# h = 5
# L = [(2,2,-2),(4,4,-1),(1,1,1),(3,3,2)]
# p = 0.9
# r = -0.25
# V, policy = value_iteration(w, h, L, p, r)
# print("test #7:")
# print("Value Function:")
# V_fliped = np.flipud(V)
# print(V_fliped)

#t8

# w = 7
# h = 7
# L = [(1,1,-4),(1,5,-6),(5,1,1),(5,5,4)]
# p = 0.8
# r = -0.5
# V, policy = value_iteration(w, h, L, p, r)
# print("test #8:")
# print("Value Function:")
# V_fliped = np.flipud(V)
# print(V_fliped)
# #t9

# w = 7
# h = 7
# L = [(1,1,-4),(1,5,-6),(5,1,1),(5,5,4)]
# p = 0.8
# r = -0.5
# V, policy = value_iteration(w, h, L, p, r)
# print("test #9:")
# print("Value Function:")
# V_fliped = np.flipud(V)
# print(V_fliped)

# w = 7
# h = 7
# L = [(3,1,0),(3,5,0),(1,1,-4),(1,5,-6),(5,1,1),(5,5,4)]
# p = 0.8
# r = -0.25
# V, policy = value_iteration(w, h, L, p, r)
# print("test #10:")
# print("Value Function:")
# V_fliped = np.flipud(V)
# print(V_fliped)
