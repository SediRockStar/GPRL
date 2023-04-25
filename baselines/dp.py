import numpy as np
import gym



def mountain_car_dynamics(s, a, min_pos=-1.2, max_pos=0.6, min_vel=-0.07, max_vel=0.07):
    position, velocity = s
    new_velocity = np.clip(velocity + 0.001 * (a- 1) - 0.0025 * np.cos(3 * position), min_vel, max_vel)
    new_position = np.clip(position + new_velocity, min_pos, max_pos)
    if new_position <= min_pos:
        new_velocity = 0
    return new_position, new_velocity

def sample_env(self, p, v):
    '''
    :param p:
    :param v:
    :return:
    '''

    r1 = bool(p >= self.env.goal_position and v >= self.env.goal_velocity)

    return r1

def discretize_state(s, position_bins, velocity_bins):
    position, velocity = s
    discretized_position = np.digitize(position, position_bins, right= True)
    discretized_velocity = np.digitize(velocity, velocity_bins, right= True)
    if position== position_bins[0]:
        discretized_position = 0
    if velocity == velocity_bins[0]:
        discretized_velocity = 0
    return discretized_position, discretized_velocity

def value_iteration(env, position_bins, velocity_bins, actions, value_function, T= 10, threshold=1e-5, gamma=0.9):
    for t in range(T):
        delta = 0
        for pos_idx, position in reversed(list(enumerate(position_bins))):
            for vel_idx, velocity in reversed(list(enumerate(velocity_bins))):
                env.reset()
                s = (position, velocity)
                old_value = value_function[pos_idx, vel_idx]
                max_action_value = float('-inf')
                for a in actions:
                    env.env.state = s
                    s_prime, reward, done, _ = env.step(int(a))
                    disc_s_prime, disc_v_prime = discretize_state(s_prime, position_bins, velocity_bins)

                    reward = 1 if s_prime[0]>= 0.5 and s_prime[1]>= env.goal_velocity else 0
                    future_return = gamma * value_function[disc_s_prime, disc_v_prime]
                    action_value = reward + future_return
                    max_action_value = max(max_action_value, action_value)
                value_function[pos_idx, vel_idx] = max_action_value
                delta = max(delta, abs(old_value - value_function[pos_idx, vel_idx]))
        if delta < threshold:
            break
    return value_function


def test_agent(value_function, position_bins, velocity_bins, gamma= 0.9):
    env.reset()
    rewards= 0
    s= env.env.state
    for i in range(200):
        #env.render()
        pos_idx, vel_idx = discretize_state(s, position_bins, velocity_bins)
        original_pos, original_vel = s
        values= []
        for a_prime in range(3):
            s_p= mountain_car_dynamics(s, a_prime)
            pos_idx_prime, vel_idx_prime = discretize_state(s_p, position_bins, velocity_bins)
            reward = 1 if s_p[0] >= 0.5 and s_p[1] >= env.goal_velocity else 0
            values.append(reward+ gamma* value_function[pos_idx_prime, vel_idx_prime])
            env.env.state = (original_pos, original_vel)
        a = np.argmax(values)
        s_prime, reward, done, _ = env.step(a)
        s = s_prime
        rewards += reward
        if done:
            break
    return rewards

# Create MountainCar-v0 environment
env = gym.make('MountainCar-v0')

# Parameters
num_positions, num_velocities = 21, 21
min_pos, max_pos = -1.2, 0.6
min_vel, max_vel = -0.07, 0.07

# Discretize state space
position_bins = np.linspace(min_pos, max_pos, num_positions)
velocity_bins = np.linspace(min_vel, max_vel, num_velocities)

# Initialize value function
value_function = np.zeros((num_positions, num_velocities))

# Define action space
actions = np.array([0, 1, 2])

# Perform value iteration
value_function = value_iteration(env, position_bins, velocity_bins, actions, value_function, T= 1)
#print(value_function)
"""
for i in range(len(value_function)):
    for j in range(len(value_function[0])):
        if(value_function[i][j]==0):
            print(i, j)

"""

# Extract optimal policy

for i in range(10):
    rewards= test_agent(value_function, position_bins, velocity_bins)
    #print(rewards)


min_pos, max_pos = -1.2, 0.6
min_vel, max_vel = -0.07, 0.07

# Discretize state space
sample_pos = np.linspace(min_pos, max_pos, 5)
sample_v = np.linspace(min_vel, max_vel, 5)

for pos in sample_pos:
    for v in sample_v:
        ds= discretize_state((pos, v), position_bins, velocity_bins)
        print(value_function[ds[0], ds[1]])

print("///////////////////////////////")