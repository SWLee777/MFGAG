import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNMemoryNetwork(nn.Module):
    def __init__(self, input_dim, emb_dim, domain_num, memory_num=10):
        super(GCNMemoryNetwork, self).__init__()
        self.domain_num = domain_num  # Number of domains
        self.emb_dim = emb_dim  # Embedding dimension
        self.memory_num = memory_num  # Number of memory slots
        self.gcn = GCNConv(input_dim, emb_dim)  # Initialize GCN layer
        # Initialize memory for each domain with zeros
        self.domain_memory = {i: torch.zeros((memory_num, emb_dim)).cuda() for i in range(domain_num)}

    def forward(self, feature, category, edge_index):
        feature = norm(feature)  # Normalize feature input
        domain_memory = [self.domain_memory[i] for i in range(self.domain_num)]  # Get domain memories
        sep_domain_embedding = []  # List to hold embeddings for each domain

        for i in range(self.domain_num):
            gcn_output = self.gcn(feature, edge_index)  # Apply GCN to the feature
            # Compute the domain-specific embedding
            tmp_domain_embedding = torch.mm(torch.softmax(torch.mm(gcn_output, domain_memory[i].T), dim=1),
                                            domain_memory[i])
            sep_domain_embedding.append(tmp_domain_embedding.unsqueeze(1))  # Append embedding for the domain

        # Concatenate embeddings from all domains
        sep_domain_embedding = torch.cat(sep_domain_embedding, 1)
        # Compute domain attention
        domain_att = torch.bmm(sep_domain_embedding, feature.unsqueeze(2)).squeeze()
        domain_att = torch.softmax(domain_att, dim=1).unsqueeze(1)  # Apply softmax to get attention weights

        return domain_att  # Return domain attention

    def write(self, all_feature, category, edge_index):
        domain_fea_dict = {}  # Dictionary to hold features for each domain
        domain_set = set(category.cpu().detach().numpy().tolist())  # Unique categories
        for i in domain_set:
            domain_fea_dict[i] = []  # Initialize list for each domain

        # Populate the domain feature dictionary
        for i in range(all_feature.size(0)):
            domain_fea_dict[category[i].item()].append(all_feature[i].view(1, -1))

        # Update memory for each domain
        for i in domain_set:
            domain_fea_dict[i] = torch.cat(domain_fea_dict[i], 0)  # Concatenate features for the domain
            gcn_output = self.gcn(domain_fea_dict[i], edge_index)  # Apply GCN to domain features
            # Update the memory using a weighted average
            new_mem = torch.mm(torch.softmax(torch.mm(gcn_output, self.domain_memory[i].T), dim=1), gcn_output)
            self.domain_memory[i] = self.domain_memory[i] * 0.95 + new_mem * 0.05  # Update memory with new values
