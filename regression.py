import torch
from torch import nn

def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    Hint: use nn.Linear
    """
    model = nn.Linear(input_size, output_size)
    return model

def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def fit_regression_model(X, y):
    """
    Train the model for the given number of epochs.
    Hint: use the train_iteration function.
    Hint 2: while working you can use the print function to print the loss every 1000 epochs.
    Hint 3: you can use the previous_loss variable to stop the training when the loss is not changing much.
    """
    learning_rate = 0.001  # Adjust the learning rate
    num_epochs = 10000  # Adjust the number of epochs
    input_features = X.shape[1]  # extract the number of features from the input `shape` of X
    output_features = y.shape[1]  # extract the number of features from the output `shape` of y
    model = create_linear_regression_model(input_features, output_features)

    loss_fn = nn.MSELoss()  # Use mean squared error loss, like in class

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previous_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

        # Change this condition to stop the training when the loss is not changing much.
        if abs(previous_loss - loss.item()) < 1e-5:
            print(f"Stopping early at epoch {epoch} due to small change in loss.")
            break

        previous_loss = loss.item()

    return model, loss

# Example usage:
# Assuming X and y are your input and output tensors, you can call the function like this:
# trained_model, final_loss = fit_regression_model(X, y)
