import dataset
from model import CharRNN, CharLSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# Import your dataset class
from dataset import Shakespeare

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function """
    model.train()
    trn_loss = 0.0

    for inputs, targets in tqdm(trn_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        hidden = model.init_hidden(inputs.size(0))
        outputs, hidden = model(inputs, hidden)
        
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        trn_loss += loss.item() * inputs.size(0)

    trn_loss = trn_loss / len(trn_loader.dataset)
    return trn_loss

def validate(model, val_loader, device, criterion):
    """ Validate function """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            hidden = model.init_hidden(inputs.size(0))

            outputs, hidden = model(inputs, hidden)

            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            val_loss += loss.item() * inputs.size(0)

    val_loss = val_loss / len(val_loader.dataset)
    return val_loss

def main():
    """ Main function """
    # Load dataset
    input_file = 'shakespeare_train.txt'  # Replace with the path to your Shakespeare text file
    dataset = Shakespeare(input_file)

    # Define hyperparameters
    input_size = len(dataset.char2idx)
    hidden_size = 128
    output_size = input_size
    num_layers = 1
    batch_size = 64
    epochs = 50
    learning_rate = 0.003
    validation_split = 0.2

    # Create data loaders with SubsetRandomSampler
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize models, optimizers, and loss function
    rnn_model = CharRNN(input_size, hidden_size, output_size, num_layers).to(device)
    lstm_model = CharLSTM(input_size, hidden_size, output_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=learning_rate)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)

    rnn_train_losses = []
    rnn_val_losses = []
    lstm_train_losses = []
    lstm_val_losses = []
    
    best_rnn_val_loss = float('inf')
    best_lstm_val_loss = float('inf')
    best_rnn_model = None
    best_lstm_model = None

    for epoch in range(epochs):
        rnn_train_loss = train(rnn_model, train_loader, device, criterion, rnn_optimizer)
        rnn_val_loss = validate(rnn_model, val_loader, device, criterion)
        lstm_train_loss = train(lstm_model, train_loader, device, criterion, lstm_optimizer)
        lstm_val_loss = validate(lstm_model, val_loader, device, criterion)

        rnn_train_losses.append(rnn_train_loss)
        rnn_val_losses.append(rnn_val_loss)
        lstm_train_losses.append(lstm_train_loss)
        lstm_val_losses.append(lstm_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, RNN Train Loss: {rnn_train_loss:.4f}, RNN Val Loss: {rnn_val_loss:.4f}")
        print(f"Epoch {epoch+1}/{epochs}, LSTM Train Loss: {lstm_train_loss:.4f}, LSTM Val Loss: {lstm_val_loss:.4f}")
        
        if rnn_val_loss < best_rnn_val_loss:
            best_rnn_val_loss = rnn_val_loss
            best_rnn_model = rnn_model.state_dict()
        
        if lstm_val_loss < best_lstm_val_loss:
            best_lstm_val_loss = lstm_val_loss
            best_lstm_model = lstm_model.state_dict()


    torch.save(best_rnn_model, 'best_rnn.pth')
    torch.save(best_lstm_model, 'best_lstm.pth')

    # Plot the losses
    plt.figure(figsize=(12, 6))
    plt.plot(rnn_train_losses, label='RNN Train Loss')
    plt.plot(rnn_val_losses, label='RNN Validation Loss')
    plt.plot(lstm_train_losses, label='LSTM Train Loss')
    plt.plot(lstm_val_losses, label='LSTM Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses for RNN and LSTM')
    plt.show()

if __name__ == '__main__':
    main()