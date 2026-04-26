import numpy as np
import time
from minitorch.tensor import Tensor
import minitorch.nn as nn
import minitorch.optim as optim
from minitorch.dataset import load_mnist

class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Flattened 28x28 = 784 -> 128 -> 10 classes
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.l1(x).relu()
        return self.l2(x)

def train():
    print("Loading MNIST...")
    X_train, Y_train = load_mnist('training', 'data')
    X_test, Y_test = load_mnist('testing', 'data')
    
    # Normalize images to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # One-hot encode labels
    def to_one_hot(y, num_classes=10):
        one_hot = np.zeros((y.size, num_classes), dtype=np.float32)
        one_hot[np.arange(y.size), y] = 1.0
        return one_hot
        
    Y_train_oh = to_one_hot(Y_train)
    Y_test_oh = to_one_hot(Y_test)

    model = MNIST_MLP()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    batch_size = 128
    epochs = 3
    num_batches = len(X_train) // batch_size

    print(f"Training on {len(X_train)} images")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        
        # Shuffle
        indices = np.random.permutation(len(X_train))
        
        start_time = time.time()
        for i in range(num_batches):
            batch_idx = indices[i*batch_size : (i+1)*batch_size]
            
            x_b = Tensor(X_train[batch_idx])
            y_b = Tensor(Y_train_oh[batch_idx])
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(x_b)
            loss = criterion(logits, y_b)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
            
            # Accuracy
            preds = np.argmax(logits.data, axis=1)
            correct += np.sum(preds == Y_train[batch_idx])
            
            if i % 100 == 0:
                print(f"Epoch {epoch} | Batch {i}/{num_batches} | Loss: {loss.data.item():.4f}")
                
        end_time = time.time()
        train_acc = correct / len(X_train)
        print(f"--- Epoch {epoch} complete in {end_time - start_time:.2f}s | Train Acc: {train_acc*100:.2f}% | Avg Loss: {total_loss/num_batches:.4f} ---")
        
        # Eval
        model.eval()
        test_logits = model(Tensor(X_test))
        test_preds = np.argmax(test_logits.data, axis=1)
        test_acc = np.sum(test_preds == Y_test) / len(Y_test)
        print(f"--- Test Accuracy: {test_acc*100:.2f}% ---\n")

if __name__ == "__main__":
    train()
