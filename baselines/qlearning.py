import gym
import numpy as np
import imageio

env = gym.make("MountainCar-v0")
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.9
EPISODES = 3000

SHOW_EVERY = 999

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    frames = []
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if render:
            frames.append(env.render(mode="rgb_array"))

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"Congratulation! We reached to the goal! Episode: {episode}")
            #_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state
    if render:
        print(frames[0].shape)
        imageio.mimsave(f'./{episode}.gif', frames, fps=40)


data= []
for episode in range(50):

    discrete_state = get_discrete_state(env.reset())
    frames = []
    done = False
    steps= 0
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        steps += 1

        discrete_state = new_discrete_state
    data.append(steps)

print(np.mean(data))
print(np.std(data))

min_pos, max_pos = -1.2, 0.6
min_vel, max_vel = -0.07, 0.07

# Discretize state space
sample_pos = np.linspace(min_pos, max_pos, 5)
sample_v = np.linspace(min_vel, max_vel, 5)

for pos in sample_pos:
    for v in sample_v:
        state= np.array([pos, v])
        discrete_state= get_discrete_state(state)
        values= q_table[discrete_state]
        print(np.max(values))

print("///////////////////////////////")


env.close()