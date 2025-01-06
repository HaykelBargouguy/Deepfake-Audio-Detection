import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SelfAttention(nn.Module):
    """
    Custom Self-Attention Mechanism designed to aggregate frame-level 
    features into utterance-level representations for anti-spoofing.
    """
    def __init__(self, hidden_size, mean_only=False):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.mean_only = mean_only
        
        # The learnable attention weight vector. It acts as a query to find 
        # the most informative frames in the input sequence.
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)
        init.kaiming_uniform_(self.att_weights)

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape (Batch, Sequence_Length, Hidden_Size)
        Returns:
            representations: Aggregated utterance-level feature vector.
        """
        batch_size = inputs.size(0)
        
        # Step 1: Compute attention scores using batch matrix multiplication.
        # We align the learnable weights with the batch dimensions.
        weights = torch.bmm(
            inputs, 
            self.att_weights.permute(1, 0).unsqueeze(0).repeat(batch_size, 1, 1)
        )

        # Step 2: Apply non-linearity and softmax to get normalized attention weights.
        if inputs.size(0) == 1:
            attentions = F.softmax(torch.tanh(weights), dim=1)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            attentions = F.softmax(torch.tanh(weights.squeeze()), dim=1)
            weighted = torch.mul(inputs, attentions.unsqueeze(2).expand_as(inputs))

        # Step 3: Aggregate the sequence into a single vector representation.
        if self.mean_only:
            return weighted.sum(1)
        else:
            # Inject slight noise for regularization during std computation
            noise = 1e-5 * torch.randn(weighted.size())
            if inputs.is_cuda:
                noise = noise.to(inputs.device)
                
            avg_repr = weighted.sum(1)
            std_repr = (weighted + noise).std(1)
            
            # Concatenate mean and standard deviation for a richer representation
            representations = torch.cat((avg_repr, std_repr), 1)
            return representations