import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    linear = nn.Linear(dim, hidden_dim)
    norm1 = norm(hidden_dim)
    relu = nn.ReLU()
    dropout = nn.Dropout(drop_prob)
    linear2 = nn.Linear(hidden_dim, dim)
    norm2 = norm(dim)
    seq = nn.Sequential(
        linear,
        norm1,
        relu,
        dropout,
        linear2,
        norm2,
    )
    return nn.Sequential(nn.Residual(seq), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    linear1 = nn.Linear(dim, hidden_dim)
    relu = nn.ReLU()
    blocks = [ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)]
    linear2 = nn.Linear(hidden_dim, num_classes)
    model = nn.Sequential(
        linear1,
        relu,
        *blocks,
        linear2
    )
    return model
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()
    
    total_loss = 0.0
    total_error = 0
    total_num = 0
    
    for batch in dataloader:
        X, y = batch
        # Flatten the input from (batch_size, H, W, C) to (batch_size, H*W*C)
        X = X.reshape((X.shape[0], -1))
        logits = model(X)
        loss = nn.SoftmaxLoss()(logits, y)
        
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
        
        # Accumulate loss
        total_loss += loss.numpy() * X.shape[0]
        
        # Calculate error (number of incorrect predictions)
        predictions = np.argmax(logits.numpy(), axis=1)
        total_error += np.sum(predictions != y.numpy())
        
        total_num += X.shape[0]
    
    avg_loss = total_loss / total_num
    error_rate = total_error / total_num
    
    return error_rate, avg_loss
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # Load MNIST datasets
    train_dataset = ndl.data.MNISTDataset(
        data_dir + "/train-images-idx3-ubyte.gz",
        data_dir + "/train-labels-idx1-ubyte.gz"
    )
    test_dataset = ndl.data.MNISTDataset(
        data_dir + "/t10k-images-idx3-ubyte.gz",
        data_dir + "/t10k-labels-idx1-ubyte.gz"
    )
    
    # Create dataloaders
    train_dataloader = ndl.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = ndl.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Initialize model
    model = MLPResNet(dim=784, hidden_dim=hidden_dim)
    
    # Initialize optimizer
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    for _ in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt)
    
    # Final evaluation
    test_error, test_loss = epoch(test_dataloader, model, opt=None)
    
    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
