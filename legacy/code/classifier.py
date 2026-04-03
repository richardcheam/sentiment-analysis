import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, output_dim=2):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # output_dim=2 for CrossEntropyLoss

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out    
    
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=2, dropout=0.3, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = x.unsqueeze(1)  # Add sequence length dimension

        _, (hn, _) = self.lstm(x)
        
        # Concatenate last hidden states from both directions
        if self.bidirectional:
            out = torch.cat((hn[-2], hn[-1]), dim=1)  # (batch_size, hidden_dim * 2)
        else:
            out = hn[-1]  # (batch_size, hidden_dim)

        out = self.fc(out)  # (batch_size, output_dim)
        return out

class DeepLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, output_dim=2, dropout=0.4, bidirectional=True, fc_hidden_dim=64):
        super(DeepLSTMClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Fully connected classifier head
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, output_dim)
        )

    def forward(self, x):
        # Input shape: (batch_size, input_dim)
        x = x.unsqueeze(1)  # Shape becomes (batch_size, seq_len=1, input_dim)

        # LSTM forward
        _, (hn, _) = self.lstm(x)  # hn: (num_layers * num_directions, batch, hidden_dim)

        # Take the last layer's hidden state from both directions
        if self.bidirectional:
            out = torch.cat((hn[-2], hn[-1]), dim=1)  # (batch_size, hidden_dim * 2)
        else:
            out = hn[-1]  # (batch_size, hidden_dim)

        # FC Classifier
        out = self.fc_layers(out)  # (batch_size, output_dim)
        return out

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):  # lstm_output: (batch, seq_len, hidden_dim)
        scores = self.attn(lstm_output).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_output)  # (batch, 1, hidden_dim)
        return context.squeeze(1)  # (batch, hidden_dim)

class LSTMWithAttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=2, dropout=0.4, bidirectional=True, fc_hidden_dim=64):
        super(LSTMWithAttentionClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.attention = Attention(hidden_dim * self.num_directions)

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, seq_len=1, input_dim)

        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * num_directions)
        context = self.attention(lstm_out)  # (batch, hidden_dim * num_directions)
        out = self.fc_layers(context)
        return out


## MoE
from transformers import DistilBertModel
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.gate(x)

class MoE(nn.Module):
    def __init__(self, num_experts=3, expert_hidden=128, output_dim=2):
        super(MoE, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.bert.requires_grad_(False)  # freeze BERT

        self.input_dim = self.bert.config.hidden_size
        self.experts = nn.ModuleList([
            Expert(self.input_dim, expert_hidden, output_dim)
            for _ in range(num_experts)
        ])
        self.gate = GatingNetwork(self.input_dim, num_experts)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # CLS token

        gate_weights = self.gate(bert_output)  # (batch_size, num_experts)

        expert_outputs = torch.stack([expert(bert_output) for expert in self.experts], dim=1)  # (batch_size, num_experts, output_dim)

        # Weighted sum of expert outputs
        weighted_output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)  # (batch_size, output_dim)

        return weighted_output
