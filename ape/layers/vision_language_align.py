import math

import torch
import torch.nn.functional as F
from torch import nn


class VisionLanguageAlign(nn.Module):
    def __init__(
        self, embed_dim, embed_dim_language, prior_prob=0.01, log_scale=0.0, clamp_dot_product=True, need_log_scale=True, return_all=False
    ):
        super().__init__()
        # initialize the bias for focal loss
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        # dot product soft token head
        self.dot_product_projection_image = nn.Identity()
        self.dot_product_projection_text = nn.Linear(
            embed_dim_language, embed_dim, bias=True
        )  # 768 -> 256
        self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)
        self.bias_lang = nn.Parameter(torch.zeros(embed_dim_language), requires_grad=True)  # (768，)
        self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)  # size (1,)
        self.need_log_scale = need_log_scale
        self.clamp_dot_product = clamp_dot_product
        self.return_all = return_all

    def forward(self, x, embedding, lvl=None):
        """
        x: visual features (bs, num_query, 256)
        embedding: language features (bs, L, 768)
        """
        # print('vl,', x.shape, embedding.shape, self.bias_lang.data.shape)
        # wadaw
        # print(x.shape,embedding.shape)
        # awdwa
        embedding = embedding.to(x.dtype)

        # norm
        embedding = F.normalize(embedding, p=2, dim=-1)  # (bs, L, 768) L is maximum sentence length
        # print(embedding.dtype,(embedding / 2.0).dtype)
        # dwadwa
        dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)  # 768 -> 256
        dot_product_proj_tokens_bias = (
            torch.matmul(embedding, self.bias_lang) + self.bias0
        )  # (bs, L, 768) x (768, ) + (1, ) -> (bs, L)

        dot_product_proj_queries = self.dot_product_projection_image(x)  # (bs, num_query, 256)
        A = dot_product_proj_queries.shape[1]  # num_query
        bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)  # (bs, num_query, L)
        # print(torch.unique(dot_product_proj_queries),torch.unique(dot_product_proj_tokens),bias)
        # awdwad

        if self.need_log_scale:
            dot_product_logit = (
                torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2))
                / self.log_scale.exp()
            ) + bias  # (bs, num_query, 256) x (bs, 256, L) -> (bs, num_query, L)

        # if lvl == 5 or lvl is None: # DEBUG
        #     print(torch.unique(embedding),torch.unique(bias),torch.unique(x),torch.unique(dot_product_logit),lvl)
        #     awdaw

        else:
            dot_product_logit = (
                torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2))
            ) + bias  # (bs, num_query, 256) x (bs, 256, L) -> (bs, num_query, L)
        # print(bias.shape,dot_product_proj_queries.shape,'111')
        if self.clamp_dot_product:
            dot_product_logit = torch.clamp(dot_product_logit, max=50000)
            dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
        if  self.return_all:
            return dot_product_proj_queries, dot_product_proj_tokens, bias
        else:
            return dot_product_logit

    def for_no_batch(self, x, embedding):
        """
        x: visual features (num_query, 256)
        embedding: language features (L, 768)
        """
        print('vl,', x.shape, embedding.shape, self.bias_lang.data.shape)
        embedding = embedding.to(x.dtype)

        # norm
        embedding = F.normalize(embedding, p=2, dim=-1)
        dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)
        dot_product_proj_tokens_bias = (
            torch.matmul(embedding, self.bias_lang) + self.bias0
        )

        dot_product_proj_queries = self.dot_product_projection_image(x)
        A = dot_product_proj_queries.shape[0]
        bias = dot_product_proj_tokens_bias.unsqueeze(0).repeat(A, 1)

        dot_product_logit = (
            torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2))
            / self.log_scale.exp()
        ) + bias
        if self.clamp_dot_product:
            dot_product_logit = torch.clamp(dot_product_logit, max=50000)
            dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
        return dot_product_logit


class StillClassifier(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.body = nn.Linear(hidden_dim, 1)

    def forward(self, x, lang_feat=None):
        return self.body(x)
