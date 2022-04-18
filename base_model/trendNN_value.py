import torch
import torch.nn as nn
# from Time2Vector import Time2Vector

class PricePredictionModel(nn.Module):
    def __init__(self, nSignal, nClasses, args):
        super(PricePredictionModel, self).__init__()

        self.device = torch.device(args.device)

        self.activation = nn.ReLU()
        self.dropout = args.dropout
        self.args = args
        # drop layer param
        self.dropLayer = nn.Dropout(p=args.dropout)

        self.softmax = nn.Softmax(dim=-1)

        # lstm param
        self.hiddenDim = args.ff_dims  # 4 inputs need 1 hidden dim
        self.nLayers = 1  # lstm hidden layer
        # self.embedding = Time2Vector(input_size=nSignal, hidden_size=self.hiddenDim)
        self.lstmInput = nSignal
        self.lstm = nn.LSTM(self.lstmInput, self.hiddenDim, self.nLayers, batch_first=True)

        # linear layers
        self.fc1 = nn.Linear(self.hiddenDim, 256)
        self.fcOut = nn.Linear(256, nClasses)

    def forward(self, state):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.nLayers, state.size(0), self.hiddenDim).to(self.device)
        # Initialize cell state
        c0 = torch.zeros(self.nLayers, state.size(0), self.hiddenDim).to(self.device)
        
        # print(f'state {state.shape}, h0 {h0.shape}  c0 {c0.shape}')
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
        state = torch.from_numpy(state).float().to(self.device)
        out = self.forwardLinear(state)
        out = out.squeeze(0)

        return out