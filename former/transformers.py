import torch
from torch import nn
import torch.nn.functional as F

from .modules import TransformerBlock

from .util import d

class GTransformer(nn.Module):
    """
    Transformer for generating text (character by character).
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, attention_type='default', split_size=4):

        super().__init__()

        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=(seq_length * 2 - 1 if attention_type=='relative' else seq_length))

        tblocks = []
        for _ in range(depth // split_size):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True, attention_type=attention_type, pos_embedding=self.pos_embedding))
            
        self.layers = nn.ModuleList([nn.Sequential(*tblocks) for _ in range(split_size)])
        self.toprobs = nn.ModuleList([nn.Linear(emb, num_tokens) for _ in range(split_size)])

    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions

        out = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            layer_logits = self.toprobs[i](x.view(b*t, e)).view(b, t, self.num_tokens)
            out.append(layer_logits)

        return tuple(out)

