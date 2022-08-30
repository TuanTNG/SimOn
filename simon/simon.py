import math
from torch.nn import Transformer
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    # helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)
                        * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(1), :][None, ...])


class TokenEmbedding(nn.Module):
    # helper Module to convert tensor of input indices into corresponding tensor of token embeddings
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class SimOn(nn.Module):
    def __init__(self, args):
        super(SimOn, self).__init__()

        input_history_feature_dim = args.dim_feature
        dropout = args.dropout
        num_decoder_layers = args.num_decoder_layers
        nhead = args.nhead
        feat_forward_dim = args.feat_forward_dim
        d_model = args.d_model

        self.transformer = Transformer(d_model=d_model,
                                       nhead=nhead,
                                       num_encoder_layers=6,  # Dummy, not use encoder
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=feat_forward_dim,
                                       dropout=dropout,
                                       batch_first=True)

        self.transformer = self.transformer.decoder

        self.generator = nn.Linear(d_model, args.numclass)

        self.feature_start_emb = TokenEmbedding(1, d_model)

        self.embed_positional_encoding = PositionalEncoding(
            d_model, dropout=dropout, maxlen=5000)

        self.linear_encoding = nn.Linear(input_history_feature_dim, d_model)

        self.linear_prob = nn.Linear(args.numclass-1, d_model)

        self.linear_encoding2 = nn.Linear(d_model*2, d_model)

    def forward(self,
                camera_inputs,
                is_starts,
                probs):
        camera_inputs = self.linear_encoding(camera_inputs)
        is_starts_num = is_starts.sum()

        if is_starts_num > 0:
            # start embedding E
            is_starts = is_starts.view(-1)
            feature_start_emb = self.feature_start_emb(
                torch.tensor([0]).cuda())
            camera_inputs[is_starts, 0:1] = \
                feature_start_emb[None, None, -1]

        N, BA, C = camera_inputs.shape

        # postional encoding PE
        camera_inputs = self.embed_positional_encoding(camera_inputs)

        queries = camera_inputs[:, -1:]

        past_frame_features = camera_inputs[:, :-1]

        encoder_prob = probs.to(torch.float32)

        # probject prob
        prob_embded = self.linear_prob(encoder_prob)

        # concat ft and ct
        memory = torch.cat(
            (past_frame_features, prob_embded), -1).view(N,
                                                         -1, prob_embded.shape[-1]*2)
        # project to lower dimmension
        memory = self.linear_encoding2(memory)
        
        # mask
        target_mask = self.generate_square_subsequent_mask(
            queries.size(1), queries.device)
       
        # input transformer
        outs = self.transformer(queries, memory, target_mask)
        # predict score
        
        cls_outputs = self.generator(outs.view(N, 1, -1))
        return cls_outputs

    @staticmethod
    def generate_square_subsequent_mask(sz, DEVICE):
        mask = (torch.triu(torch.ones((sz, sz), device=DEVICE))
                == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask
