import torch
import torch.nn as nn
from Time2Vector import Time2Vector

class PricePredictionModel(nn.Module):
    def __init__(self, nSignal, nClasses, device, dropout, embedding=True, hiddenDim=64):
        super(PricePredictionModel, self).__init__()

        self.device = device

        self.activation = nn.ReLU()

        # drop layer param
        self.dropLayer = nn.Dropout(p=dropout)

        self.softmax = nn.Softmax(dim=-1)

        # lstm param
        self.hiddenDim = hiddenDim  # 4 inputs need 1 hidden dim
        self.nLayers = 1  # lstm hidden layer
        self.embed = embedding
        self.embedding = Time2Vector(input_size=nSignal, hidden_size=self.hiddenDim)
        self.lstmInput = self.hiddenDim if embedding else nSignal
        self.lstm = nn.LSTM(self.lstmInput, self.hiddenDim, self.nLayers, batch_first=True, dropout = dropout)

        # linear layers
        self.fc1 = nn.Linear(self.hiddenDim, 256)
        self.fcOut = nn.Linear(256, nClasses)

    def forward(self, state):
        if self.embed:
            state = torch.squeeze(state,0)
            state = self.embedding(state)
            state = state.unsqueeze(0)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.nLayers, state.size(0), self.hiddenDim).to(self.device)
        # Initialize cell state
        c0 = torch.zeros(self.nLayers, state.size(0), self.hiddenDim).to(self.device)
        out, (hn, cn) = self.lstm(state, (h0, c0))

        # Index hidden state of last time step
        # Read out last time step hidden states
        out = out[:, -1, :]
        # out = out.squeeze()[-1, :]

        return out

    def forwardLinear(self, state):
        out = self.forward(state)
        out = self.dropLayer(self.activation(self.fc1(out)))
        out = self.fcOut(out)

        return out

    def selectClass(self, state):
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        out = self.forwardLinear(state)
        out = out.squeeze(0)

        return out