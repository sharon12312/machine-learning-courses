import numpy as np
import pylab as plt
import networkx as nx
import pandas as pd

# helper functions
def get_available_actions(state):
    current_state_row = R[state,]
    available_actions = np.where(current_state_row >= 0)[1]
    return available_actions

def sample_next_action(available_actions):
    next_action = int(np.random.choice(available_actions, size=1))
    return next_action

def update(current_state, action, gamma):
    max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1]
    print('max index: ', max_index.shape)

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)

    max_value = Q[action, max_index]
    Q[current_state, action] = R[current_state, action] + gamma * max_value  # preform a mathematical formula
    print('max value: ', R[current_state, action] + gamma * max_value)


# initial variables
edge_list = [(0, 2), (0, 1), (0, 3), (2, 4), (5, 6), (7, 4), (0, 6), (5, 3), (3, 7), (0, 8)]
goal = 7

# networkx library help us to create a visualized Graph and presents it
G = nx.Graph()
G.add_edges_from(edge_list)
position = nx.spring_layout(G)

nx.draw_networkx_nodes(G, position)
nx.draw_networkx_edges(G, position)
nx.draw_networkx_labels(G, position)
# plt.show()

# create a matrix
SIZE_MATRIX = 9
R = np.matrix(np.ones(shape=(SIZE_MATRIX, SIZE_MATRIX)))
R *= -1  # initial values in the matrix to be -1

# case 1: the reward for any edge which leads to the goal (node 7) will be 100
# case 2: if the node is from the goal edge, it leads to the goal as well,
# as the graph is undirected
for edge in edge_list:
    print(edge)

    if edge[1] == goal:
        R[edge] = 100
    else:
        R[edge] = 0

    if edge[0] == goal:
        R[edge[::-1]] = 100
    else:
        R[edge[::-1]] = 0

R[goal, goal] = 100

print(R)

gamma = 0.8
Q = np.matrix(np.zeros([SIZE_MATRIX, SIZE_MATRIX]))
print(pd.DataFrame(Q))

# an example of 1 step
initial_state = 0
available_actions = get_available_actions(initial_state)
print(available_actions)

action = sample_next_action(available_actions)
print(action)

update(initial_state, action, gamma)

# preform he Q-value iteration algorithms for 700 iterations
# the Q values will converge to the optimal state values corresponding to the optimal policy
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_actions = get_available_actions(current_state)
    action = sample_next_action(available_actions)
    update(current_state, action, gamma)

print('Trained Q matrix: ', pd.DataFrame(Q))
print('Normalized Q matrix: ', pd.DataFrame(Q / np.max(Q) * 100))

current_state = 0
steps = [current_state]

while current_state != 7:
    next_step_index = np.where(Q[current_state, ] == np.max(Q[current_state, ]))[1]

    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)

    steps.append(next_step_index)
    current_state = next_step_index

print('Mose efficient path: ', steps)
