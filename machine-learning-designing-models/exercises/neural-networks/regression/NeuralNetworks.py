import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch
import torch.optim as optim

advertising_data = pd.read_csv('data/Advertising.csv')
advertising_data = advertising_data.drop('index', axis=1)
print(advertising_data.head())

# pre processing
# neural networks perform better when the values are small, so we need to scale the figures
advertising_data[['TV']] = preprocessing.scale(advertising_data[['TV']])
advertising_data[['radio']] = preprocessing.scale(advertising_data[['radio']])
advertising_data[['newspaper']] = preprocessing.scale(advertising_data[['newspaper']])

print(advertising_data.sample(10))

X = advertising_data.drop('sales', axis=1)
Y = advertising_data[['sales']]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# convert to tensor format
x_train_tensor = torch.tensor(x_train.values, dtype=torch.float)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)

print(x_train_tensor.shape, y_train_tensor.shape)
print(x_test_tensor.shape, y_test_tensor.shape)

# build our neural networks
inp = 3
out = 1
hid = 100
loss_fn = torch.nn.MSELoss()
learning_rate = 0.0001

model = torch.nn.Sequential(torch.nn.Linear(inp, hid),
                            torch.nn.ReLU(),
                            torch.nn.Linear(hid, out))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(10000):
    y_pred = model(x_train_tensor)
    loss = loss_fn(y_pred, y_train_tensor)

    if iter % 1000 == 0:
        print(iter, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

y_pred_tensor = model(x_test_tensor)
y_pred = y_pred_tensor.detach().numpy()
print(y_pred[:5])

# visualization
# plt.figure(figsize=(8, 8))
# plt.scatter(y_pred, y_test.values)
# plt.xlabel('Actual Sale')
# plt.ylabel('Predicted Sale')
# plt.title('Predicted Sale vs Actual Sale')
# plt.show()

# model's accuracy
print(r2_score(y_test, y_pred))
