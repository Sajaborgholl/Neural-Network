# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


def initialize_model(input_size, hidden_size, output_size):
    from src.NeuralNetwork import SimpleNeuralNetwork
    model = SimpleNeuralNetwork(input_size, hidden_size, output_size)
    return model


def train_model(model, X_train, y_train, X_val=None, y_val=None, num_epochs=100, learning_rate=0.001, weight_decay=1e-5):
    # Convert training data to tensors
    X_train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.long)

    # If validation data is provided, convert it to tensors
    if X_val is not None and y_val is not None:
        X_val_tensor = torch.tensor(np.array(X_val), dtype=torch.float32)
        y_val_tensor = torch.tensor(np.array(y_val), dtype=torch.long)

    # Define loss function and optimizer with L2 regularization (weight decay)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Lists to store loss and accuracy at each epoch
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass for training data
        outputs = model(X_train_tensor)
        train_loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimization
        train_loss.backward()
        optimizer.step()

        # Track training loss
        train_losses.append(train_loss.item())

        # Validation phase, if validation data is provided
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_losses.append(val_loss.item())

                # Calculate validation accuracy
                _, predicted = torch.max(val_outputs, 1)
                val_accuracy = accuracy_score(
                    y_val_tensor.numpy(), predicted.numpy())
                val_accuracies.append(val_accuracy)

        # Print results at each epoch
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_loss.item():.4f}, "
                  f"Validation Loss: {
                      val_loss.item() if X_val is not None else 'N/A':.4f}, "
                  f"Validation Accuracy: {val_accuracy * 100 if X_val is not None else 'N/A':.2f}%")

    return model, train_losses, val_losses, val_accuracies


def evaluate_model(model, X_test, y_test):
    # Convert to numpy if X_test or y_test are in DataFrame or Series format
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.to_numpy()
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()

    # Convert testing data to tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Model evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)

    # Calculate accuracy
    accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return accuracy
