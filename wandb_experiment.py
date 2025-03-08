import wandb

# Initialize a new run
run = wandb.init(
    entity="bt-fluidnetop",
    project="wandb-test",
    config={
        "lr": 0.01,
    }
)

# Make a NN and train it to learn XOR
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple feedforward neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Create a dataset for training
x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Create a model, loss function, and optimizer
model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=run.config.lr)

# Train the model
for epoch in range(5000):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss {loss.item()}")
    run.log({"loss": loss.item()})

run.finish()

