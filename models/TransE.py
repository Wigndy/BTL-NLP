from turtle import forward
import numpy as np
import torch
import torch.nn as nn

class TransE(nn.Module):
    def __init__(self, entity_count, relation_count, device, norm=1, dim=100, margin=1.0):
        super(TransE, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.device = device
        self.norm = norm
        self.dim = dim
        self.entities_emb = self._init_entity_emb()
        self.relations_emb = self._init_relation_emb()
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')
        
    def _init_entity_emb(self):
        entities_emb = nn.Embedding(num_embeddings=self.entity_count + 1, embedding_dim=self.dim, padding_idx=self.entity_count)
        uniform_range = 6 / np.sqrt(self.dim)
        entities_emb.weight.data.uniform_(-uniform_range, uniform_range)
        return entities_emb
    
    def _init_relation_emb(self):
        relations_emb = nn.Embedding(num_embeddings=self.relation_count + 1, embedding_dim=self.dim, padding_idx=self.relation_count)
        uniform_range = 6 / np.sqrt(self.dim)
        relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
        relations_emb.weight.data[:-1,:].div_(relations_emb.weight.data[:-1,:].norm(p=1, dim=1, keepdim=True))
        return relations_emb
    
    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
        self.entities_emb.weight.data[:-1,:].div_(self.entities_emb.weight.data[:-1,:].norm(p=2, dim=1, keepdim=True))
        assert positive_triplets.size()[1] == 3
        positive_distances = self._distance(positive_triplets)
        
        assert negative_triplets.size()[1] == 3
        negative_distances = self._distance(negative_triplets)
        
        return self.loss(positive_distances, negative_distances), positive_distances, negative_distances
    
    def predict(self, triplets: torch.LongTensor):
        """Calculated dissimilarity score for given triplets.

        :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
        :return: dissimilarity score for given triplets
        """
        return self._distance(triplets)
    
    def loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)
        
    def _distance(self, triplets):
        assert triplets.size()[1] == 3
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        return (self.entities_emb(heads) + self.relations_emb(relations) - self.entities_emb(tails)).norm(p=self.norm,dim=1)
    
            
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class TransE(nn.Module):
#     def __init__(self, n_entities, n_relations, embedding_dim=100):
#         super(TransE, self).__init__()
#         self.ent_emb = nn.Embedding(n_entities, embedding_dim)
#         self.rel_emb = nn.Embedding(n_relations, embedding_dim)
#         # Uniform init as per paper
#         sqrt_dim = embedding_dim ** 0.5
#         nn.init.uniform_(self.ent_emb.weight.data, a=-6/sqrt_dim, b=6/sqrt_dim)
#         nn.init.uniform_(self.rel_emb.weight.data, a=-6/sqrt_dim, b=6/sqrt_dim)
    
#     def forward(self, head, rel, tail):
#         h = self.ent_emb(head)
#         r = self.rel_emb(rel)
#         t = self.ent_emb(tail)
#         return torch.norm(h + r - t, p=1, dim=1)  # L1 distance




# model = TransE(n_ent, n_rel)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# gamma = 1.0  # Margin

# for epoch in range(1000):
#     for batch_h, batch_r, batch_t in train_loader:
#         optimizer.zero_grad()
#         pos_score = model(batch_h, batch_r, batch_t)
#         # Corrupt head/tail (uniform random)
#         neg_h = torch.randint(0, n_ent, batch_h.size(), device=device)
#         neg_score = model(neg_h, batch_r, batch_t)
#         loss = F.relu(gamma + pos_score - neg_score).mean()
#         loss.backward()
#         optimizer.step()
#         # Normalize entity embeddings
#         model.ent_emb.weight.data = F.normalize(model.ent_emb.weight.data, p=2, dim=1)
