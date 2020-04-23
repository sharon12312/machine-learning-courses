import gym
import numpy as np

# the observation will be the one with the maximum Q-value
def choose_action(observ):
    return np.argmax(q_table[observ])


# set the environment
env = gym.make('FrozenLake-v0')

# number of possible actions: left, right, top, bottom
print(env.action_space.n)

# describe the current state of the environment
print(env.observation_space)  # 4 x 4 = 16 (S-start, F-frozen-lake, H-hole, G-goal)

# learning rate
alpha = 0.4
# discount factor
gamma = 0.999

# initialize a Q table, every state initial to 1
q_table = dict([(x, [1, 1, 1, 1]) for x in range(16)])
print(q_table)

for i in range(10000):
    observ = env.reset()
    action = choose_action(observ)

    prev_observ = None
    prev_action = None

    t = 0
    for t in range(2500):
        env.render()
        observ, reward, done, info = env.step(action)
        action = choose_action(observ)

        if not prev_observ is None:
            q_old = q_table[prev_observ][prev_action]
            q_new = q_old

            if done:
                q_new += alpha * (reward - q_old)
            else:
                q_new += alpha * (reward + gamma * q_table[observ][action] - q_old)  # mathematical formula

            # update the state table for the previous action with he new values calculated ater the current action
            new_table = q_table[prev_observ]
            new_table[prev_action] = q_new

            q_table[prev_observ] = new_table

        prev_observ = observ
        prev_action = action

        if done:
            print('Episode {} finished after {} time steps with r={}.'.format(i, t, reward))
            break