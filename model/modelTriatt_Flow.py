import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Encoder_overall(Module):
    """\
    Overall encoder (scheme 3: dual-LLM branches + InfoNCE contrastive learning).
    """
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1,
                 dim_in_feat_omics2, dim_out_feat_omics2,
                 dim_emb=256,
                 dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        # self.dim_in_feat_omics1: RNA feature dimension
        # self.dim_in_feat_omics2: protein feature dimension
        # self.dim_out_feat_omics1: RNA output dimension
        # self.dim_out_feat_omics2: protein output dimension
        # self.dim_emb: embedding dimension
        # self.dropout: dropout rate
        # self.act: activation function
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dim_emb = dim_emb
        self.dropout = dropout
        self.act = act

        
        # dim_emb: embedding dimension 512
        # dim_out_feat_omics1: RNA output dimension
        self.encoder_embedding1 = Cell_embedding_Encoder(self.dim_emb, self.dim_out_feat_omics1)
        # dim_out_feat_omics1: RNA output dimension
        # dim_emb: embedding dimension 512
        self.decoder_embedding1 = Cell_embedding_Decoder(self.dim_out_feat_omics1, self.dim_emb)

        # scGPT branch:
        # dim_emb: embedding dimension
        # dim_out_feat_omics1: RNA output dimension
        self.encoder_embedding2 = Cell_embedding_Encoder(self.dim_emb, self.dim_out_feat_omics1)
        # dim_out_feat_omics1: RNA output dimension
        # dim_emb: embedding dimension
        self.decoder_embedding2 = Cell_embedding_Decoder(self.dim_out_feat_omics1, self.dim_emb)

        # additive attention fusing the two LLM spa branches
        # self.dim_out_feat_omics1: RNA output dimension
        # self.dim_out_feat_omics1: RNA output dimension
        self.atten_llm_spa = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        # additive attention fusing the two LLM fea branches
        self.atten_llm_fea = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)

        # RNA modality
        # self.dim_in_feat_omics1: RNA feature dimension
        # self.dim_out_feat_omics1: RNA output dimension
        self.encoder_omics1 = DeepEncoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        # self.dim_out_feat_omics1: RNA output dimension
        # self.dim_in_feat_omics1: RNA feature dimension
        self.decoder_omics1 = DeepDecoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)

        # ADT modality
        # self.dim_in_feat_omics2: protein feature dimension
        # self.dim_out_feat_omics2: protein output dimension
        self.encoder_omics2 = DeepEncoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        # self.dim_out_feat_omics2: protein output dimension
        # self.dim_in_feat_omics2: protein feature dimension
        self.decoder_omics2 = DeepDecoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)

        # attention layers
        self.atten_feature1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_feature2 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_feature  = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_omics2   = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        self.atten_cross    = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2)



                # features_omics1: RNA features
                # features_omics2: protein features
                # adj_spatial_omics1: RNA spatial graph
                #adj_feature_omics1: RNA feature graph
                # adj_spatial_omics2: protein spatial graph
                # adj_feature_omics2: protein feature graph
                # embedding_genef: Geneformer embedding
                # embedding_scgpt: scGPT embedding
                # adj_emb_genef: Geneformer embedding graph
                # adj_emb_scgpt: scGPT embedding graph

    def forward(self, features_omics1, features_omics2,
                adj_spatial_omics1, adj_feature_omics1,
                adj_spatial_omics2, adj_feature_omics2,
                embedding_genef, embedding_scgpt,
                adj_emb_genef, adj_emb_scgpt):

        # ========== dual-LLM branches: fully independent GCNs ==========
        # Geneformer spa branch
        # embedding_genef: Geneformer embedding
        # adj_spatial_omics1: RNA spatial graph
        emb_genef_spa = self.encoder_embedding1(embedding_genef, adj_spatial_omics1)
        # Geneformer fea branch
        # embedding_genef: Geneformer embedding
        # adj_emb_genef: Geneformer embedding graph
        emb_genef_fea = self.encoder_embedding1(embedding_genef, adj_emb_genef)

        # scGPT spa branch
        emb_scgpt_spa = self.encoder_embedding2(embedding_scgpt, adj_spatial_omics1)
        # scGPT fea branch
        emb_scgpt_fea = self.encoder_embedding2(embedding_scgpt, adj_emb_scgpt)

        # ========== AttentionLayer fuses the two LLMs (per branch) ==========
        #  Geneformer attention weights
        #  scGPT attention weights. For visualisation: alpha_llm_spa (attention weights
        #  of the two LLM spatial branches), alpha_llm_fea (attention weights of the two
        #  LLM feature branches)
        # emb_llm_spa: fusion of the two LLM spatial graphs
        emb_llm_spa, alpha_llm_spa = self.atten_llm_spa(emb_genef_spa, emb_scgpt_spa)
        # emb_llm_fea: fusion of the two LLM feature graphs
        emb_llm_fea, alpha_llm_fea = self.atten_llm_fea(emb_genef_fea, emb_scgpt_fea)

        # ========== RNA modality encoding ==========
        # features_omics1: RNA features
        # adj_spatial_omics1: RNA spatial graph
        # emb_latent_spatial_omics1: RNA spatial graph fusion result
        emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial_omics1)
        # features_omics1: RNA features
        # adj_feature_omics1: RNA feature graph
        # emb_latent_feature_omics1: RNA feature graph fusion result
        emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)

        # ========== fuse the LLM result with the RNA result ==========
        # emb_llm_spa: LLM spatial graph fusion result
        # emb_latent_spatial_omics1: RNA spatial graph fusion result
        # !!! for visualisation: alpha_att1 (fusion weights of the LLM spatial result and
        # the RNA spatial result)

        # embedding_att1 (fusion of the LLM spatial result and the RNA spatial result)
        embedding_att1, alpha_att1 = self.atten_feature1(emb_llm_spa, emb_latent_spatial_omics1)

        # emb_llm_fea: fusion of the two LLM feature graphs
        # emb_latent_feature_omics1: RNA feature graph fusion result
        # !!! for visualisation: alpha_att2 (fusion weights of the two-LLM feature result
        # and the RNA feature result)
        # embedding_att2 (fusion of the LLM feature result and the RNA feature result)
        embedding_att2, alpha_att2 = self.atten_feature2(emb_llm_fea, emb_latent_feature_omics1)

        # embedding_att1: fusion of the LLM spatial result and the RNA spatial result
        # embedding_att2: fusion of the two-LLM feature result and the RNA feature result
        # embedding_latent_omics1:
        # embedding_latent_omics1 is another weighted fusion of the spatial view and the
        # feature view; it is the unified RNA representation after dual-path
        # (spatial + feature) and dual-level (LLM + raw) fusion.
        # alpha_att_omics1: attention weights over embedding_att1 (spatial branch) and
        # embedding_att2 (feature branch), of shape (n_cells, 2).

        
        embedding_latent_omics1, alpha_att_omics1 = self.atten_feature(embedding_att1, embedding_att2)

        # ========== omics2 ==========
        # features_omics2: protein features
        # adj_spatial_omics2: protein spatial graph
        # emb_latent_spatial_omics2: protein spatial graph fusion result
        emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial_omics2)
        # features_omics2: protein features
        # adj_feature_omics2: protein feature graph
        # emb_latent_feature_omics2: protein feature graph fusion result
        emb_latent_feature_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)


        # emb_latent_spatial_omics2: protein spatial graph fusion result
        # emb_latent_feature_omics2: protein feature graph fusion result
        # emb_latent_omics2: fusion of the protein spatial result and the protein feature result
        # alpha_omics2: fusion weights of the protein spatial result and the protein feature result
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2,
                                                            emb_latent_feature_omics2)

        # embedding_latent_omics1: another weighted fusion of the spatial view and the
        # feature view; the unified RNA representation after dual-path (spatial +
        # feature) and dual-level (LLM + raw) fusion.
        # emb_latent_omics2: fusion of the protein spatial result and the protein feature result

        # alpha_cross: cross-modality attention weights over the RNA fusion result
        # embedding_latent_omics1 and the protein fusion result emb_latent_omics2, of
        # shape (n_cells, 2).
        #  alpha_cross[:, 0]: attention weight of each cell on the RNA modality
        #  alpha_cross[:, 1]: attention weight of each cell on the protein modality
        emb_latent_combined, alpha_cross = self.atten_cross(embedding_latent_omics1,
                                                           emb_latent_omics2)
 
        # ========== decoders ==========
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)

        # each LLM reconstructs its own spa branch (fix 1: use the LLM's own feature
        # graph, symmetric with the spa encoder)
        emb_recon_genef = self.decoder_embedding1(emb_genef_spa, adj_spatial_omics1)
        emb_recon_scgpt = self.decoder_embedding2(emb_scgpt_spa, adj_spatial_omics1)

        # each LLM reconstructs its own fea branch (removes the blind spot with no
        # direct supervision)
        emb_recon_genef_fea = self.decoder_embedding1(emb_genef_fea, adj_emb_genef)
        emb_recon_scgpt_fea = self.decoder_embedding2(emb_scgpt_fea, adj_emb_scgpt)

        # ========== correspondence ==========
        emb_latent_omics1_across_recon = self.encoder_omics2(
            self.decoder_omics2(embedding_latent_omics1, adj_spatial_omics2), adj_spatial_omics2)
        emb_latent_omics2_across_recon = self.encoder_omics1(
            self.decoder_omics1(emb_latent_omics2, adj_spatial_omics1), adj_spatial_omics1)

        # ========== for the CL loss: the "unfused" latent of the spa branch ==========
        results = {
            'emb_latent_omics1': embedding_latent_omics1,
            'emb_latent_omics2': emb_latent_omics2,
            'emb_latent_combined': emb_latent_combined,
            'emb_recon_omics1': emb_recon_omics1,
            'emb_recon_omics2': emb_recon_omics2,
            'emb_recon_genef': emb_recon_genef,
            'emb_recon_scgpt': emb_recon_scgpt,
            'emb_recon_genef_fea': emb_recon_genef_fea,
            'emb_recon_scgpt_fea': emb_recon_scgpt_fea,
            'emb_latent_omics1_across_recon': emb_latent_omics1_across_recon,
            'emb_latent_omics2_across_recon': emb_latent_omics2_across_recon,
            'raw_emb_genef': embedding_genef,
            'raw_emb_scgpt': embedding_scgpt,
            'emb_genef_spa_raw': emb_genef_spa,
            'emb_scgpt_spa_raw': emb_scgpt_spa,
            'alpha_omics1': alpha_att_omics1,
            'alpha_omics2': alpha_omics2,
            'alpha': alpha_cross, # visualisation: cross-modality attention weights of the RNA fusion result embedding_latent_omics1 and the protein fusion result emb_latent_omics2, of shape (n_cells, 2).
            'alpha_att1': alpha_att1, # visualisation: (fusion of the LLM spatial result and the RNA spatial result)
            'alpha_att2': alpha_att2,# visualisation: (fusion of the LLM feature result and the RNA feature result)
            'alpha_llm_spa': alpha_llm_spa,
            'alpha_llm_fea': alpha_llm_fea,
        }

        return results


class Encoder(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x
'''
DeepEncoder, line by line
A multi-layer graph convolutional encoder (Deep Graph Convolutional Encoder) that
applies deep graph convolution to the raw RNA / protein features. It has a sandwich
structure: linear transform -> graph convolution -> activation -> dropout, repeated
three times.
'''

class DeepEncoder(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(DeepEncoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        self.hidden_dim = out_feat * 2
        self.weight1 = Parameter(torch.FloatTensor(self.in_feat, self.hidden_dim))
        self.weight3 = Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))
        self.weight2 = Parameter(torch.FloatTensor(self.hidden_dim, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight3)

    def forward(self, feat, adj):
        # step 1: linear transform
        # torch.mm is matrix multiplication
        # input: (n_cells, in_feat) x (in_feat, hidden_dim) = (n_cells, hidden_dim)
        # effect: project each cell embedding from the `in_feat` space to `hidden_dim`
        x = torch.mm(feat, self.weight1)
        # step 2: graph convolution message passing
        # torch.spmm is sparse matrix x dense matrix multiplication
        # input: (n_cells, n_cells) x (n_cells, hidden_dim) = (n_cells, hidden_dim)
        # effect: sum the neighbour features of every cell, weighted
        x = torch.spmm(adj, x)
        # step 3: activation
        x = self.act(x)
        # step 4: dropout
        x = F.dropout(x, self.dropout, training=self.training)
        # step 5: linear transform
        # torch.mm is matrix multiplication
        # input: (n_cells, hidden_dim) x (hidden_dim, hidden_dim) = (n_cells, hidden_dim)
        # effect: project each cell embedding from `hidden_dim` to `hidden_dim`
        x = torch.mm(x, self.weight3)
        # step 6: graph convolution message passing
        # torch.spmm is sparse matrix x dense matrix multiplication
        # input: (n_cells, n_cells) x (n_cells, hidden_dim) = (n_cells, hidden_dim)
        # effect: sum the neighbour features of every cell, weighted
        x = torch.spmm(adj, x)
        # step 7: activation
        x = self.act(x)
        x = F.dropout(x, self.dropout, training=self.training)
        # step 8: linear transform
        # torch.mm is matrix multiplication
        # input: (n_cells, hidden_dim) x (hidden_dim, out_feat) = (n_cells, out_feat)
        # effect: project each cell embedding from `hidden_dim` to `out_feat`
        x = torch.mm(x, self.weight2)
        # step 9: graph convolution message passing
        # torch.spmm is sparse matrix x dense matrix multiplication
        # input: (n_cells, n_cells) x (n_cells, out_feat) = (n_cells, out_feat)
        # effect: sum the neighbour features of every cell, weighted
        x = torch.spmm(adj, x)
        return x


class Cell_embedding_Encoder(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Cell_embedding_Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        '''
        
        **Step 1: linear transform** (equivalent to a fully connected layer)
- `torch.mm` is matrix multiplication
- input: `(n_cells, in_feat)` x `(in_feat, out_feat)` = `(n_cells, out_feat)`
- effect: project each cell embedding from the `in_feat` space to `out_feat`
---
```x = torch.spmm(adj, x)```
**Step 2: graph convolution message passing**
- `torch.spmm` is **sparse matrix x dense matrix** multiplication (more efficient than `torch.mm`)
- `adj`: sparse adjacency matrix `(n_cells, n_cells)`; each row lists which neighbours a cell is connected to
- `x`: the output of `torch.mm`, `(n_cells, out_feat)`
- result: `(n_cells, n_cells)` x `(n_cells, out_feat)` = `(n_cells, out_feat)`
**Meaning**: for each cell $i$, sum the features of its neighbours $j \in \mathcal{N}(i)$, weighted:
        
        
        '''
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x


class Cell_embedding_Decoder(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Cell_embedding_Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x


class Decoder(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x


class DeepDecoder(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(DeepDecoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        self.hidden_dim = out_feat * 2
        self.weight1 = Parameter(torch.FloatTensor(self.in_feat, self.hidden_dim))
        self.weight3 = Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))
        self.weight2 = Parameter(torch.FloatTensor(self.hidden_dim, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight3)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight1)
        x = torch.spmm(adj, x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x, self.weight3)
        x = torch.spmm(adj, x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.mm(x, self.weight2)
        x = torch.spmm(adj, x)
        return x

# Additive attention layer that fuses two embeddings (emb1, emb2) and returns the fused
# embedding plus the attention weights. Its design follows Bahdanau attention.
class AttentionLayer(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        # in_feat: input dimension
        # out_feat: output dimension
        # dropout: dropout rate
        # act: activation function
        self.in_feat = in_feat
        self.out_feat = out_feat
        # learnable weight matrix of shape `(in_feat, out_feat)`
        # **purpose**: project each of the two input embeddings into the hidden dimension
        # `out_feat` via a linear transform. This is the core of additive attention:
        # transform `emb1` and `emb2` first, then fuse.
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        # learnable weight vector of shape `(out_feat, 1)` (a matrix whose second dimension is 1).
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = []
        # torch.squeeze(emb1): if emb1 has shape (n_cells, dim) it usually stays unchanged
        # (unless one dimension is 1). This is for compatibility with a possible
        # (n_cells, 1, dim) shape.
        # torch.unsqueeze(..., dim=1): insert a dimension at dim 1 (the column direction)
        # result: (n_cells, dim) -> (n_cells, 1, dim)
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        # concatenate the two embeddings: two (n_cells, 1, dim) tensors -> (n_cells, 2, dim)
        self.emb = torch.cat(emb, dim=1)
        # step 1: compute the attention scores (core)
        # torch.matmul(self.emb, self.w_omega): matrix multiplication
        # input: (n_cells, 2, in_feat) x (in_feat, out_feat) = (n_cells, 2, out_feat)
        # effect: linearly transform emb1 and emb2 separately into the hidden dimension out_feat
        # F.tanh(...): hyperbolic tangent activation on the transformed result (non-linearity)
        # result shape: (n_cells, 2, out_feat)
        # meaning: linearly transform + non-linearly activate the two embeddings of each cell.
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        # step 2: compress the hidden vectors into scalar scores
        # torch.matmul(self.v, self.u_omega): matrix multiplication
        # input: (n_cells, 2, out_feat) x (out_feat, 1) = (n_cells, 2, 1)
        # result shape: (n_cells, 2, 1)
        # meaning: emit one scalar score (the unnormalized attention weight) per hidden
        # vector of each cell.
        self.vu = torch.matmul(self.v, self.u_omega)
        # step 3: normalize to obtain the attention weights
        # torch.squeeze(self.vu): squeeze (n_cells, 2, 1) into (n_cells, 2)
        # F.softmax(...): softmax over dim 1 (i.e. between the two embeddings)
        # result shape: (n_cells, 2)
        # meaning: softmax-normalize the two scalar scores of each cell.
        # $\alpha_i[0]$ and $\alpha_i[1]$ are the attention weights on emb1[i] and emb2[i].
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)
        # step 4: weighted fusion of the two embeddings with the attention weights
        # torch.transpose(self.emb, 1, 2): swap dimensions
        # input: (n_cells, 2, in_feat) -> output: (n_cells, in_feat, 2)
        # now each cell is a row, the dimension is in the middle and the attention weight last
        # torch.unsqueeze(self.alpha, -1): insert a dimension at the end
        # input: (n_cells, 2) -> output: (n_cells, 2, 1)
        # becomes a column vector for the weighted sum
        # then the matrix multiplication: (n_cells, in_feat, 2) x (n_cells, 2, 1) is not a
        # standard matrix product
        # this is in fact a batch matmul: each cell independently computes
        # (in_feat, 2) x (2, 1) = (in_feat, 1)
        # final shape: (n_cells, in_feat, 1)
        # meaning: weighted fusion of the two embeddings with the attention weights
        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2),
                                    torch.unsqueeze(self.alpha, -1))
        # step 5: return the fused embedding and the attention weights
        # torch.squeeze(emb_combined): squeeze (n_cells, 2, 1)
        # self.alpha: attention weights
        # return value: (n_cells, out_feat) fused embedding and (n_cells, 2) attention weights
        return torch.squeeze(emb_combined), self.alpha


class AttentionLayer_tri(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer_tri, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2, cell_emb):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(cell_emb), dim=1))
        self.emb = torch.cat(emb, dim=1)
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)
        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2),
                                    torch.unsqueeze(self.alpha, -1))
        return torch.squeeze(emb_combined), self.alpha


class AttentionLayer_spa_fea(Module):
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer_spa_fea, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2, cell_emb_spa, cell_emb_fea):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(cell_emb_spa), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(cell_emb_fea), dim=1))
        self.emb = torch.cat(emb, dim=1)
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)
        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2),
                                    torch.unsqueeze(self.alpha, -1))
        return torch.squeeze(emb_combined), self.alpha
