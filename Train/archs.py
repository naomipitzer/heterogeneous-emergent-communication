import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np


class Sender(nn.Module):
    """Sender Network"""
    
    def __init__(self, feat_dim, h_dim, w_dim, bin_dim_out, use_binary=True):
        super(Sender, self).__init__()
        self.feat_dim = feat_dim  # Input feature dimension (128 for VGGish embeddings)
        self.h_dim = h_dim  # Hidden dimension size
        self.w_dim = w_dim  # Message input dimension
        self.bin_dim_out = bin_dim_out  # Output message size
        self.use_binary = use_binary  # Whether output is binary
        
        # Linear layers to process inputs
        self.audio_layer = nn.Linear(self.feat_dim, self.h_dim)
        self.code_layer = nn.Linear(self.w_dim, self.h_dim)
        self.binary_layer = nn.Linear(self.h_dim, self.bin_dim_out)

        self.h_x = None
        self.h_w = None
        
        # Bias for the output message
        self.code_bias = nn.Parameter(torch.Tensor(self.bin_dim_out))
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_normal_(self.audio_layer.weight)
        nn.init.xavier_normal_(self.code_layer.weight)
        nn.init.xavier_normal_(self.binary_layer.weight)
        self.audio_layer.bias.data.zero_()
        self.code_layer.bias.data.zero_()
        self.binary_layer.bias.data.zero_()
        self.code_bias.data.normal_()
    
    def forward(self, x, w):
        """Generate message based on input audio features and receiver message.
        
        Args:
            x: Audio feature vector (batch_size, 128)
            w: Receiver's message vector (batch_size, w_dim)
        
        Returns:
            features: Output message (binary or continuous)
            feature_probs: Probabilities for binary output (if applicable)
        """

        self.h_x = self.audio_layer(x)  # Transform audio features
        self.h_w = self.code_layer(w)   # Transform receiver's message
        
        # Combine features using element-wise sum
        features = self.binary_layer(F.tanh(self.h_x + self.h_w))
        
        if self.use_binary:
            # Binary output using sigmoid function
            probs = torch.sigmoid(features)

            # During training: sample binary message from Bernoulli distribution
            if self.training:
                probs_ = probs.data.cpu().numpy()
                binary_features = torch.from_numpy(
                    (np.random.rand(*probs_.shape) < probs_).astype('float32'))
            # During testing: select most likely message
            else:
                binary_features = torch.round(probs).detach()

            if probs.is_cuda:
                binary_features = binary_features.cuda()

            return binary_features, probs
        else:
            return features, None


class Receiver(nn.Module):
    """Agent 2 Network: Receiver"""
    
    def __init__(self, z_dim, desc_dim, hid_dim, out_dim, w_dim, s_dim, use_binary):
        super(Receiver, self).__init__()
        self.z_dim = z_dim  # Input message dimension
        self.desc_dim = desc_dim  # Distractors input dimension
        self.hid_dim = hid_dim  # Hidden state dimension
        self.out_dim = out_dim  # Output prediction dimension
        self.w_dim = w_dim  # Output message dimension
        self.s_dim = s_dim  # Stop bit dimension
        self.use_binary = use_binary  # Whether to use binary messages

        # RNN network for updating hidden state
        self.rnn = nn.GRUCell(self.z_dim, self.hid_dim)
        
        # Layers for Receiver communications
        self.w_h = nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.w_d = nn.Linear(self.desc_dim, self.hid_dim, bias=False)
        self.w = nn.Linear(self.hid_dim, self.w_dim)
        self.h_z = None
        self.s_prob_prod = None
        
        # Layers for Receiver predictions
        self.y1 = nn.Linear(self.hid_dim + self.desc_dim, self.hid_dim)
        self.y2 = nn.Linear(self.hid_dim, self.out_dim)
        
        # Layer for Receiver stop decision
        self.s = nn.Linear(self.hid_dim, self.s_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GRUCell):
                for param in m.parameters():
                    if param.ndimension() == 2:
                        nn.init.xavier_normal_(param)
                    elif param.ndimension() == 1:
                        param.data.zero_()

    def reset_state(self):
        """Initialize state for Receiver."""
        self.h_z = None
        self.s_prob_prod = None

    def initial_state(self, batch_size):
        """Create an initial hidden state for the RNN."""
        return torch.zeros(batch_size, self.hid_dim)

    def forward(self, z, desc):
        """Process incoming communication and make predictions.
        
        Args:
            z: Communication input (batch_size, z_dim).
            desc: Image input (batch_size, desc_dim). TODO - wrong, needs to be (batch_size, n_classes,desc_dim)
        
        Returns:
            s_binary, s_prob: Stop bit decision and probability.
            w_feats, w_probs: Communication output and probabilities.
            y: Predictions.
        """
        batch_size, _ = z.size()
        
        # Initialize hidden state if necessary
        if self.h_z is None:
            self.h_z = self.initial_state(batch_size)
        
        # Update hidden state with RNN
        self.h_z = self.rnn(z, self.h_z)
        
        # Combine hidden state with description for predictions
        inp_with_desc = build_inp(self.h_z, desc)  # B*D x (WV+h)

        # Compute stop probability
        s_score = self.s(self.h_z)
        s_prob = torch.sigmoid(s_score)
        s_binary = torch.round(s_prob).detach() if not self.training else torch.bernoulli(s_prob)
        if self.training:
            s_binary = torch.bernoulli(s_prob)
        else: # Infer decisions deterministically
            if self.s_prob_prod is None:
                self.s_prob_prod = s_prob
            else:
                self.s_prob_prod *= s_prob
            s_binary = torch.round(self.s_prob_prod).detach() # Round the probability to get a binary decision

        # Obtain predictions
        y = self.y1(inp_with_desc).clamp(min=0)
        y = self.y2(y)
        # Alternative approach using reshape
        y = y.view(batch_size, -1)
        # output dimension needs to be 1!!!!!

        # Obtain communications
        n_desc = y.size(1)  # Number of descriptions
        # Reweight descriptions based on current model confidence
        y_scores = torch.softmax(y, dim=1).detach()
        # Expand y_scores to match the description dimension
        y_broadcast = y_scores.unsqueeze(-1).expand(-1, n_desc, self.desc_dim)

        # Expand description vectors to match batch size
        wd_inp = desc.expand(batch_size, n_desc, self.desc_dim)
        wd_inp = (y_broadcast * wd_inp).sum(1).squeeze(1)

        # Hidden state for Receiver message
        
        self.h_w = F.tanh(self.w_h(self.h_z) + self.w_d(wd_inp))
        w_scores = self.w(self.h_w)
        
        if self.use_binary:
            w_probs = torch.sigmoid(w_scores)
            w_binary = torch.round(w_probs).detach() if not self.training else torch.bernoulli(w_probs).detach()
            w_feats = w_binary
        else:
            w_feats = w_scores
            w_probs = None
            
        return (s_binary, s_prob), (w_feats, w_probs), y


class Baseline(nn.Module):
    """Baseline model to estimate agent's loss based on its input features."""

    def __init__(self, hid_dim, x_dim, binary_dim, inp_dim):
        super(Baseline, self).__init__()
        self.x_dim = x_dim
        self.binary_dim = binary_dim
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim

        # Define two fully connected layers
        self.linear1 = nn.Linear(x_dim + self.binary_dim + self.inp_dim, self.hid_dim)
        self.linear2 = nn.Linear(self.hid_dim, 1)  # Output is a scalar score

    def forward(self, x, binary, inp):
        """Estimate agent's loss based on the agent's input.

        Args:
            x: Image features (or other type of input).
            binary: Communication message (binary vector).
            inp: Hidden state (used when the agent is the Receiver).
        
        Output:
            pred_score: A scalar estimate of the agent's loss or performance.
        """
        features = []
        if x is not None:
            features.append(x)
        if binary is not None:
            features.append(binary)
        if inp is not None:
            features.append(inp)
        
        # Concatenate the input features
        features = torch.cat(features, 1)

        # Pass through the first linear layer (ReLU activation)
        hidden = self.linear1(features).clamp(min=0)

        # Pass through the second linear layer to get the final score
        pred_score = self.linear2(hidden)
        return pred_score


#------------------- Functions -------------------#

def build_inp(binary_features, descs):
    """Function preparing input for Receiver network

    Args:
        binary_features: List of communication vectors, length ``B``.
        descs: List of description vectors, length ``D``.
    Output:
        b_cat_d: The cartesian product of binary features and descriptions, length ``B`` x ``D``.
    """
    if descs is not None:
        batch_size, num_desc, desc_dim = descs.shape
        _, feature_dim = binary_features.shape

        # Expand binary features.
        binary_copied = binary_features.unsqueeze(1).expand(batch_size, num_desc, feature_dim)
        inp = torch.cat([binary_copied, descs], dim=2)

        # Flatten the input to (batch_size * num_desc, feature_dim + desc_dim)
        inp = inp.view(batch_size * num_desc, feature_dim + desc_dim)
        return inp
    else:
        return binary_features
