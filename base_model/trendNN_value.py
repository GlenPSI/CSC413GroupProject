import torch
import torch.nn as nn
# from Time2Vector import Time2Vector

class PricePredictionModel(nn.Module):
    def __init__(self, nSignal, nClasses, args):
        super(PricePredictionModel, self).__init__()

        self.device = torch.device(args.device)

        self.activation = nn.ReLU()
        self.dropout = args.dropout if args.nLayers > 1 else 0
        self.args = args
        # drop layer param
        self.dropLayer = nn.Dropout(p=args.dropout)

        self.softmax = nn.Softmax(dim=-1)

        # lstm param
        self.hiddenDim = args.ff_dims  # 4 inputs need 1 hidden dim
        self.nLayers = args.nLayers  # lstm hidden layer
        # self.embedding = Time2Vector(input_size=nSignal, hidden_size=self.hiddenDim)
        self.lstmInput = nSignal
        self.lstm = nn.LSTM(self.lstmInput, self.hiddenDim, self.nLayers, batch_first=True, dropout=self.dropout)

        # linear layers
        self.fc1 = nn.Linear(self.hiddenDim, 256)
        self.fcOut = nn.Linear(256, nClasses)

        self.ffOut = nn.Linear(self.hiddenDim, nClasses)

    def forward(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.nLayers, state.size(0), self.hiddenDim).to(self.device)
        # Initialize cell state
        c0 = torch.zeros(self.nLayers, state.size(0), self.hiddenDim).to(self.device)
        
        out, (hn, cn) = self.lstm(state, (h0, c0))

        # Index hidden state of last time step
        # Read out last time step hidden states
        out = out[:, -1, :]
        # out = out.squeeze()[-1, :]
        # out = self.activation(self.fc1(out))
        # out = self.dropLayer(self.fcOut(out))
        # out = self.dropLayer(self.ffOut(out))
        out = self.ffOut(out)
        out = out.squeeze(1)

        # out = self.dropLayer(self.activation(self.fc1(out)))
        # out = self.fcOut(out)
        # out = out.squeeze(0)
        return out

    def forwardLinear(self, state):
        out = self.forward(state)

        return out

    def selectClass(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        out = self.forwardLinear(state)
        out = out.squeeze(0)

        return out



class Model(nn.Module):
    def __init__(self, input_size, output_size, args):
        super(Model, self).__init__()
        self.device = torch.device(args.device)
        self.input_size = input_size
        self.hidden_size = args.ff_dims
        self.output_size = output_size
        self.lstm = nn.LSTM(self.lstmInput, self.hiddenDim, self.nLayers, batch_first=True, dropout=args.dropout)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.nLayers = args.nLayers 

    def forward(self, state, future=0):
        outputs = []
        state = torch.from_numpy(state).float().to(self.device)
        #reset the state of LSTM
        #the state is kept till the end of the sequence
        h_t = torch.zeros(self.nLayers, state.size(0), self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(self.nLayers, state.size(0), self.hidden_size, dtype=torch.float32)

        for i, input_t in enumerate(state.chunk(state.size(1), dim=1)):
            h_t, c_t = self.lstm(input_t, (h_t,c_t))
            output = self.linear(h_t)
            outputs += [output]

        for i in range(future):
            h_t, c_t = self.lstm(output,(h_t,c_t))
            output = self.linear(h_t)
            outputs += [output]
            outputs = torch.stack(outputs,1).squeeze(2)
        return outputs  
