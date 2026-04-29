import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn as nn

_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

class AnomalyCLIP_PromptLearner(nn.Module):
    # 增加 num_layers 参数，默认 4 层
    def __init__(self, clip_model, design_details, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        classnames = ["object"]
        self.n_cls = len(classnames)
        self.n_ctx = design_details["Prompt_length"]
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"] 
        dtype = clip_model.transformer.get_cast_dtype()
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.classnames = classnames
        self.state_normal_list = ["{}"]
        self.state_anomaly_list = ["damaged {}"]
        
        normal_num = len(self.state_normal_list)
        anormaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num

        print(f"Initializing class-specific contexts for {num_layers} independent layers")
        # 为 4 个层分别初始化独立的上下文向量，不加 layer_weights
        ctx_vectors_pos = torch.empty(num_layers, self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype)
        ctx_vectors_neg = torch.empty(num_layers, self.n_cls, self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors_pos, std=0.02)
        nn.init.normal_(ctx_vectors_neg, std=0.02)

        prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
        prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
        
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        prompts_pos = [prompt_prefix_pos +  " " + template.format(name)+ "." for template in self.state_normal_list for name in classnames]
        prompts_neg = [prompt_prefix_neg +  " " + template.format(name)+ "." for template in self.state_anomaly_list for name in classnames]

        tokenized_prompts_pos = torch.cat([tokenize(p) for p in prompts_pos])
        tokenized_prompts_neg = torch.cat([tokenize(p) for p in prompts_neg])
     
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            _, l, d = embedding_pos.shape
            embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)

        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :] )
        self.register_buffer("token_suffix_pos", embedding_pos[:, :,1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:,:, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])

        _, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)
        _, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)

    def forward(self, cls_id=None):
        prefix_pos = self.token_prefix_pos.unsqueeze(0).expand(self.num_layers, -1, -1, -1, -1)
        suffix_pos = self.token_suffix_pos.unsqueeze(0).expand(self.num_layers, -1, -1, -1, -1)
        
        prompts_pos = torch.cat([prefix_pos, self.ctx_pos, suffix_pos], dim=3)
        prompts_pos = prompts_pos.reshape(self.num_layers, -1, prompts_pos.shape[-2], prompts_pos.shape[-1])

        prefix_neg = self.token_prefix_neg.unsqueeze(0).expand(self.num_layers, -1, -1, -1, -1)
        suffix_neg = self.token_suffix_neg.unsqueeze(0).expand(self.num_layers, -1, -1, -1, -1)
        
        prompts_neg = torch.cat([prefix_neg, self.ctx_neg, suffix_neg], dim=3)
        prompts_neg = prompts_neg.reshape(self.num_layers, -1, prompts_neg.shape[-2], prompts_neg.shape[-1])

        prompts = torch.cat([prompts_pos, prompts_neg], dim=1)
        prompts = prompts.reshape(self.num_layers * 2, prompts.shape[-2], prompts.shape[-1])

        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1, self.tokenized_prompts_pos.shape[-1])
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1, self.tokenized_prompts_neg.shape[-1])
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim=0)
        tokenized_prompts = tokenized_prompts.unsqueeze(0).expand(self.num_layers, -1, -1).reshape(self.num_layers * 2, -1)

        return prompts, tokenized_prompts, self.compound_prompts_text