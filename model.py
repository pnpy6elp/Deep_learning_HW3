import torch.nn as nn
import torch

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        # write your codes here
        super(CharRNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        

    def forward(self, input, hidden):

        # write your codes here
        embed = self.embed(input)
        output, hidden = self.rnn(embed, hidden)
        output = self.fc(output)
        

        return output, hidden

    def init_hidden(self, batch_size):

        # write your codes here
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        initial_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        return initial_hidden


class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CharLSTM, self).__init__()

        # write your codes here
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):

        # write your codes here

        embed = self.embed(input)

        output, hidden = self.lstm(embed, hidden)
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size):

        # write your codes here
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        initial_hidden = (h0, c0)

        return initial_hidden
