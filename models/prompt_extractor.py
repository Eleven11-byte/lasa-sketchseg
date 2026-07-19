import clip
import torch
from torch import nn
from typing import List, Tuple


class PromptExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._buffer_init = False
        self.with_trainable_params = False

    def init_buffer(self, clip_model):
        self._buffer_init = True

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        raise NotImplementedError

class LearnablePromptExtractor(PromptExtractor):
    def __init__(self, prompt_dim: int, prompt_shape: Tuple[int, int]):
        """
        prompt_shape = (n_prefix, n_suffix)
        """
        super().__init__()
        self.prompt_dim = prompt_dim
        self.prompt_shape = prompt_shape

        self.prefix_prompt = self._init_prompt(self.n_prefix)
        self.suffix_prompt = self._init_prompt(self.n_suffix)

        self.with_trainable_params = True
        self._buffer_init = False
        self.noun_bucket = {}

    def _init_prompt(self, length):
        if length == 0:
            return None
        p = nn.Parameter(torch.empty(length, self.prompt_dim))
        nn.init.normal_(p, std=0.02)
        return p

    def init_buffer(self, clip_model):
        """
        初始化 CLIP 的 token embedding 作为固定信号
        """
        sentence = "X."
        tokens = clip.tokenize(sentence)
        with torch.no_grad():
            tokens = tokens.to(clip_model.token_embedding.weight.device)
            embedding = clip_model.token_embedding(tokens).type(clip_model.dtype)

        self.register_buffer("start_signal", embedding[0, :1])
        self.register_buffer("dot_signal", embedding[0, 2:3])
        self.register_buffer("end_signal", embedding[0, 3:4])
        self.register_buffer("pad_signal", embedding[0, 4:5])

        self._buffer_init = True

    @property
    def n_prefix(self):
        return self.prompt_shape[0]

    @property
    def n_suffix(self):
        return self.prompt_shape[1]

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        if not self._buffer_init:
            raise RuntimeError("PromptExtractor buffer not initialized")

        self._update_noun_features(noun_list, clip_model)

        prefix = [self.start_signal]
        if self.prefix_prompt is not None:
            prefix.append(self.prefix_prompt)
        prefix = torch.cat(prefix)

        suffix = []
        if self.suffix_prompt is not None:
            suffix.append(self.suffix_prompt)
        suffix.extend([self.dot_signal, self.end_signal])
        suffix = torch.cat(suffix)

        embeddings = []
        indices = []

        for noun in noun_list:
            noun_embed = self.noun_bucket[noun]
            tokens = torch.cat([prefix, noun_embed, suffix])
            length = tokens.shape[0]

            if length < 77:
                pad = self.pad_signal.expand(77 - length, -1)
                tokens = torch.cat([tokens, pad])

            embeddings.append(tokens)
            indices.append(length - 1)

        embeddings = torch.stack(embeddings)          # [C, 77, D]
        indices = torch.tensor(indices).to(embeddings.device)

        text_features = self._encode_text(embeddings, indices, clip_model)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def _update_noun_features(self, noun_list, clip_model):
        new_words = [n for n in noun_list if n not in self.noun_bucket]
        if len(new_words) == 0:
            return

        with torch.no_grad():
            # tokens, lengths = clip.tokenize(new_words, return_length=True)
            tokens = clip.tokenize(new_words)#.to(device)
            lengths = (tokens == 49407).int().argmax(dim=1)
            embeds = clip_model.token_embedding(tokens.to(self.device)).type(clip_model.dtype)

            for w, emb, l in zip(new_words, embeds, lengths):
                self.noun_bucket[w] = emb[1:1 + (l - 2)]

    @staticmethod
    def _encode_text(x, indices, clip_model):
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = clip_model.ln_final(x).type(clip_model.dtype)
        x = x[torch.arange(x.shape[0]), indices] @ clip_model.text_projection
        return x

    @property
    def device(self):
        return self.start_signal.device
