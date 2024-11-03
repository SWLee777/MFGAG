import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math


from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.alpha = alpha
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# Define a multi-layer perceptron (MLP) model
class MLP(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))  # Fully connected layer
            layers.append(torch.nn.BatchNorm1d(embed_dim))  # Batch normalization
            layers.append(torch.nn.ReLU())  # Activation function
            layers.append(torch.nn.Dropout(p=dropout))  # Dropout layer
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))  # Output layer
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Adjust one-dimensional tensor to two-dimensional
        return self.mlp(x)

# Define CNN feature extractor
class cnn_extractor2(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor2, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(input_size, feature_num, kernel),
                nn.BatchNorm1d(feature_num),
                nn.ReLU(),
                nn.Conv1d(feature_num, feature_num, 1),  # Increase depth
                nn.BatchNorm1d(feature_num),
                nn.ReLU()
            ) for kernel, feature_num in feature_kernel.items()]
        )
        self.input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        self.residual_conv = nn.Conv1d(input_size, self.input_shape, 1)  # Residual connection adjustment

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)  # (batch_size, 768, 170)
        residual = self.residual_conv(share_input_data)  # (batch_size, 320, 170)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]  # (batch_size, 64, 1)
        feature = torch.cat(feature, dim=1)  # (batch_size, 320, 1)
        residual = torch.max_pool1d(residual, residual.shape[-1])  # (batch_size, 320, 1)
        feature = feature + residual  # (batch_size, 320, 1)
        feature = feature.view(input_data.size(0), -1)  # (batch_size, 320)
        return feature

# Define CNN feature extractor
class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.blocks = nn.ModuleList(
            [self._make_block(input_size, feature_num, kernel) for kernel, feature_num in feature_kernel.items()]
        )
        self.input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        self.residual_conv = nn.Conv1d(input_size, self.input_shape, kernel_size=1)  # Residual connection adjustment

    def _make_block(self, input_size, feature_num, kernel):
        return nn.Sequential(
            nn.Conv1d(input_size, feature_num, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(feature_num),
            nn.ReLU(),
            nn.Conv1d(feature_num, feature_num, kernel_size=3, padding=1),  # Increase depth
            nn.BatchNorm1d(feature_num),
            nn.ReLU(),
            nn.Conv1d(feature_num, feature_num, kernel_size=3, padding=1),  # Increase depth
            nn.BatchNorm1d(feature_num),
            nn.ReLU()
        )

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)  # (batch_size, channels, seq_length)
        residual = self.residual_conv(share_input_data)  # (batch_size, input_shape, seq_length)

        feature = []
        for block in self.blocks:
            f = block(share_input_data)  # (batch_size, feature_num, seq_length)
            f = torch.max_pool1d(f, f.shape[-1])  # (batch_size, feature_num, 1)
            feature.append(f)

        feature = torch.cat(feature, dim=1)  # (batch_size, input_shape, 1)
        residual = torch.max_pool1d(residual, residual.shape[-1])  # (batch_size, input_shape, 1)

        feature = feature + residual  # (batch_size, input_shape, 1)
        feature = feature.view(input_data.size(0), -1)  # (batch_size, input_shape)
        return feature

# Define masked attention mechanism
class MaskAttention(torch.nn.Module):
    """
    Compute attention layer
    """
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)  # Attention layer

    def forward(self, inputs, mask=None):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))  # Compute attention scores
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))  # Masking
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)  # Softmax processing
        outputs = torch.matmul(scores, inputs).squeeze(1)  # Attention-weighted sum
        return outputs, scores

# Define standard attention mechanism
class Attention(torch.nn.Module):
    """
    Compute 'Scaled Dot Product Attention'
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))  # Compute attention scores
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))  # Masking
        p_attn = F.softmax(scores, dim=-1)  # Softmax processing
        if dropout is not None:
            p_attn = dropout(p_attn)  # Dropout processing
        return torch.matmul(p_attn, value), p_attn

# Define multi-headed attention mechanism
class MultiHeadedAttention(torch.nn.Module):
    """
    Take in model size and number of heads.
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h  # Dimension of each head
        self.h = h  # Number of heads

        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])  # Linear transformation layers
        self.output_linear = torch.nn.Linear(d_model, d_model)  # Output layer
        self.attention = Attention()  # Attention layer
        self.dropout = nn.Dropout(p=dropout)  # Dropout layer

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)  # Repeat mask
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]  # Linear transformations

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)  # Compute attention

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)  # Reassemble

        return self.output_linear(x), attn

# Define self-attention feature extraction module
class SelfAttentionFeatureExtract(torch.nn.Module):
    def __init__(self, multi_head_num, input_size, output_size):
        super(SelfAttentionFeatureExtract, self).__init__()
        self.attention = MultiHeadedAttention(multi_head_num, input_size)  # Multi-head attention layer
        self.out_layer = torch.nn.Linear(input_size, output_size)  # Output layer

    def forward(self, inputs, query, mask=None):
        mask = mask.view(mask.size(0), 1, 1, mask.size(-1))  # Adjust mask shape

        feature, attn = self.attention(query=query,
                                        value=inputs,
                                        key=inputs,
                                        mask=mask)  # Compute attention
        feature = feature.contiguous().view([-1, feature.size(-1)])  # Adjust feature shape
        out = self.out_layer(feature)  # Through output layer
        return out, attn
