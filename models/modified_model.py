from collections import OrderedDict

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as ttf
from PIL import Image
import math
from functools import reduce
from operator import mul
import copy
from models.prompt_extractor import LearnablePromptExtractor


class ModifiedCLIP(nn.Module):
    def __init__(self, cfg, device):
        super(ModifiedCLIP, self).__init__()
        self.cfg = cfg
        self.device = device
        self.attn_type = cfg.attn_type
        self.fuse_feature = cfg.fuse_feature
        self.size = cfg.size
        self.select_layer = cfg.select_layer
        self.mode = cfg.mode
        self.visualize_custom_attn = True  # 添加可视化控制参数
        self.visualize_layer_attn =  False  # 添加可视化不同层注意力的控制参数
        self.img = None
        self.img_name = None

        # 模型加载
        if cfg.model_name == "ViT-B/16":
            # model_path = "/home/xiaoyi/.cache/clip/ViT-B-16.pt"
            model, preprocess = clip.load("ViT-B/16", device=device)
        elif cfg.model_name == "ViT-L/14":
            # model_path = "/home/xiaoyi/.cache/clip/ViT-L-14.pt"
            model, preprocess = clip.load("ViT-L/14", device=device)
        # TODO：这里的模型加载应该还需要改
        else:
            model_name = cfg.model_name
            print(model_name)
            print(cfg.checkpoint_path)
            # NotImplementedError(f"Error: model name {cfg.model_name} not implemented")
            model, preprocess = clip.load("ViT-B/16", device=device)
            state_dict = torch.load(cfg.checkpoint_path)

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                tmp = k[:5]
                if k[:5] == "model":
                    name = k[6:]  # remove `model.`
                else:
                    name = k
                new_state_dict[name] = v

            load_info = model.load_state_dict(new_state_dict, strict=False)
            print(load_info)
            print("trained model loaded")
            # NotImplementedError(f"Error: model name {model_name} not implemented")

        # model, preprocess = clip.load(model_path, device=device)

        # NOTE：训练时不应该在eval模式
        # self.model = model.eval()
        if self.mode == "train":
            self.model = model
        elif self.mode == "test":
            self.model = model.eval()

        self.preprocess = ttf.Compose([self._resize] + preprocess.transforms[2:])
        self.patch_size = int(cfg.model_name.split("/")[-1])  # TODO: 后面改模型这里要改
        self.logit_scale = nn.Parameter(torch.tensor(cfg.logit_scale), requires_grad=False)

        self.layers = model.visual.transformer.layers

        # NOTE:加入用于跨模态蒸馏的分支
        self.use_distill = cfg.train.use_distill
        if self.use_distill:
            self.visual_encoder_copy = copy.deepcopy(model.visual)
            # 可选：冻结拷贝后的参数
            for param in self.visual_encoder_copy.parameters():
                param.requires_grad = False

        # NOTE：修改了forward函数
        self.modify()

        self.img_h, self.img_w = None, None
        self.attn = None
        self.img_part_features = None
        self.image_feature = []

        self.templates = cfg.semantic_templates
        # self.add_template = cfg.add_template

        # NOTE: visual prompt相关设置
        if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            self.prompt_config = cfg.MODEL.PROMPT
            self.add_prompt = True
            self.prompt_dropout = torch.nn.Dropout(self.prompt_config.DROPOUT)
            num_tokens = self.prompt_config.NUM_TOKENS
            self.num_tokens = num_tokens
            # if project the prompt embeddings
            if self.prompt_config.PROJECT > -1:
                # only for prepend / add
                prompt_dim = self.prompt_config.PROJECT
                self.prompt_proj = nn.Linear(
                    prompt_dim, 768)
                nn.init.kaiming_normal_(
                    self.prompt_proj.weight, a=0, mode='fan_out')
            else:
                prompt_dim = 768
                self.prompt_proj = nn.Identity()

            # initiate prompt:
            if self.prompt_config.INITIATION == "random":
                val = math.sqrt(6. / float(3 * reduce(mul, (self.patch_size, self.patch_size), 1) + prompt_dim))  # noqa

                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)

                if self.prompt_config.DEEP:  # noqa
                    total_d_layer = 12 - 1  # config.transformer["num_layers"]-1
                    self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                        total_d_layer, num_tokens, prompt_dim))
                    # xavier_uniform initialization
                    nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

            else:
                raise ValueError("Other initiation scheme is not supported")
        else:
            self.add_prompt = False
            self.prompt_config = None

        # NOTE:加入text prompt learning

        self.use_prompt = cfg.MODEL.USE_PROMPT

        if self.use_prompt:
            print("using prompt")
            self.prompt_extractor = LearnablePromptExtractor(
                prompt_dim=512,
                prompt_shape=(
                    cfg.MODEL.PROMPT.NUM_PREFIX,
                    cfg.MODEL.PROMPT.NUM_SUFFIX
                )
            )
            self.prompt_extractor.init_buffer(self.model)
        else:
            self.prompt_extractor = None

        if cfg.train.freeze_text:
            print("Freeze text encoder")
            for p in model.transformer.parameters():
                p.requires_grad = False

        """
        for p in self.model.token_embedding.parameters():
            p.requires_grad = False
        """
        self.model.positional_embedding.require_grad = False

    def custom_attn(self, attn_layer, x, attn_mask=None):

        num_heads = attn_layer.num_heads
        _, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if self.attn_type == "fused-attn" and attn_mask is not None:
            # sum to 1
            mask = attn_mask.float()

            # --------- normalize over -2 ---------
            denom1 = mask.sum(dim=-2, keepdim=True)
            denom1 = denom1 + (denom1 == 0).float()  # 防止除零
            mask = mask / denom1

            # --------- normalize over -1 ---------
            denom2 = mask.sum(dim=-1, keepdim=True)
            denom2 = denom2 + (denom2 == 0).float()
            mask = mask / denom2

            # --------- symmetric + zero-mean ---------
            mask = (mask + mask.transpose(-2, -1)) * 0.5
            mask = mask - mask.mean(dim=-2, keepdim=True)

            # --------- clamp (out-of-place) ---------
            mask = torch.clamp(mask, min=0.0)

            # --------- final normalize ---------
            denom3 = mask.sum(dim=-1, keepdim=True)
            denom3 = denom3 + (denom3 == 0).float()
            mask = mask / denom3

            # --------- reshape ---------
            mask = mask.flatten(0, 1)
            attn_weights = torch.repeat_interleave(mask, dim=0, repeats=v.shape[0] // mask.shape[0])
            """
            attn_mask /= torch.sum(attn_mask, dim=-2, keepdim=True)
            attn_mask /= torch.sum(attn_mask, dim=-1, keepdim=True)
            attn_mask = (attn_mask + attn_mask.transpose(-2, -1)) / 2
            attn_mask -= attn_mask.mean(-2, keepdim=True)
            attn_mask = torch.clamp(attn_mask, 0)
            attn_mask /= torch.sum(attn_mask, dim=-1, keepdim=True)

            attn_mask = attn_mask.flatten(0, 1)
            attn_weights = torch.repeat_interleave(attn_mask, dim=0, repeats=v.shape[0] // attn_mask.shape[0])
            """
        elif self.attn_type == "q-q":
            attn_weights = torch.bmm(q * scale, q.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        elif self.attn_type == "k-k":
            attn_weights = torch.bmm(k * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        elif self.attn_type == "v-v":
            attn_weights = torch.bmm(v * scale, v.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        elif self.attn_type == "vanilla":
            attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        else:
            identity = torch.eye(v.shape[-2], dtype=v.dtype, device=v.device)[None]
            attn_weights = torch.repeat_interleave(identity, dim=0, repeats=v.shape[0])

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)

        # 可视化注意力权重
        """
        if self.visualize_custom_attn:
            attn_matrix = attn_weights.mean(0).detach().cpu().numpy()
            plt.figure(figsize=(10, 8))
            plt.imshow(attn_matrix, cmap='viridis')
            plt.colorbar()
            plt.title(f'Attention Weights for {self.attn_type}')
            plt.savefig(f'attention_{self.attn_type}.png')
            plt.close()
        """
        return attn_output, attn_weights

    """
    def forward_visual_pri(self, x: torch.Tensor):
        x = self.visual_encoder_copy.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.visual_encoder_copy.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual_encoder_copy.positional_embedding.to(x.dtype)
        x = self.visual_encoder_copy.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual_encoder_copy.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])

        if self.visual_encoder_copy.proj is not None:
            x = x @ self.visual_encoder_copy.proj

        return x
    """

    def forward_visual_pri(self, x: torch.Tensor, return_layers=None):
        if return_layers is None:
            return_layers = []

        collected_feats = []

        # =========================================================
        # 1. Patch embedding (CLIP ViT standard)
        # =========================================================
        x = self.visual_encoder_copy.conv1(x)  # [B, C, H', W']
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, HW]
        x = x.permute(0, 2, 1)  # [B, HW, C]

        x = torch.cat(
            [
                self.visual_encoder_copy.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1],
                    dtype=x.dtype, device=x.device
                ),
                x
            ],
            dim=1  # [B, HW+1, C]
        )

        x = x + self.visual_encoder_copy.positional_embedding.to(x.dtype)
        x = self.visual_encoder_copy.ln_pre(x)

        x = x.permute(1, 0, 2)  # [N+1, B, D]

        for i, block in enumerate(self.visual_encoder_copy.transformer.resblocks):
            x = block(x)
            layer_idx = i + 1  # 1-based index

            if layer_idx in return_layers:
                collected_feats.append(
                    x.permute(1, 0, 2).contiguous()  # [B, N+1, D]
                )

        x = x.permute(1, 0, 2)  # [B, N+1, D]

        if self.visual_encoder_copy.proj is not None:
            x = x @ self.visual_encoder_copy.proj

        if len(return_layers) > 0:
            # [L, B, N+1, D]
            return torch.stack(collected_feats, dim=0)
        else:
            return x

    def forward_transformer_intermediate(self, x: torch.Tensor, return_layers=(7, 10), ):
        model_transformer = self.model.visual.transformer
        feats = []

        for i, block in enumerate(model_transformer.resblocks):
            x = x + block.attn(block.ln_1(x))[0]
            x = x + block.mlp(block.ln_2(x))

            layer_idx = i + 1
            if layer_idx in return_layers:
                # [N, B, 768] -> [B, N, 768]
                feats.append(x.permute(1, 0, 2).contiguous())

        return torch.stack(feats, dim=0)  # [L, B, N, 768]

    def forward_visual(self, x: torch.Tensor):
        B = x.shape[0]
        model_visual = self.model.visual
        h, w = x.shape[-2], x.shape[-1]
        positional_embedding_new = self.upsample_pos_emb(model_visual.positional_embedding,
                                                         (h // self.patch_size, w // self.patch_size))
        x = model_visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([model_visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                              dtype=x.dtype, device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + positional_embedding_new.to(x.dtype)

        if self.add_prompt:
            # ADD VISUAL PROMPTS HERE
            if self.num_tokens > 0:
                x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1)
            # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        x = model_visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        return model_visual.transformer(x)

    def forward_transformer(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        x = x.clone()
        model_transformer = self.model.visual.transformer
        model_visual = self.model.visual
        # 计算到倒数第二层
        attn_maps, img_features = 0, torch.zeros([self.layers] + list(x.shape), device=x.device, dtype=x.dtype)
        for i in range(self.layers - 1):
            ln_x = model_transformer.resblocks[i].ln_1(x)
            if self.fuse_feature or (self.select_layer + self.layers) % self.layers == i:
                img_features[i] = ln_x.clone()
            try:
                ln_x, attn_map = model_transformer.resblocks[i].attn(
                    ln_x,
                    ln_x,
                    ln_x,
                    need_weights=True,
                    attn_mask=attn_mask,
                    average_attn_weights=False,
                )
            except TypeError:
                ln_x, attn_map = model_transformer.resblocks[i].attn(
                    ln_x,
                    ln_x,
                    ln_x,
                    need_weights=True,
                    attn_mask=attn_mask,
                )
                if attn_map.dim() == 3:
                    attn_map = attn_map.unsqueeze(1)
            attn_maps += attn_map

            # 可视化每层注意力
            if self.mode == "test" and self.visualize_layer_attn:
                attn_matrix = attn_map[0].mean(0)[1:,1:] # 取第一个batch，平均heads
                """
                plt.figure(figsize=(10, 8))
                plt.imshow(attn_matrix, cmap='viridis')
                plt.colorbar()
                plt.title(f'Attention Weights for Layer {i}')
                plt.savefig(f'./attention_map/attention_layer_{i}.png')
                plt.close()
                """
                pass

            x = x + ln_x.clone()
            x = x + model_transformer.resblocks[i].mlp(model_transformer.resblocks[i].ln_2(x)).clone()

        # 计算最后一层
        model_res = model_transformer.resblocks[-1]
        img_features[-1] = x if self.fuse_feature or self.select_layer == -1 else 0
        for kth, x in enumerate(img_features):
            x_k = img_features[kth].clone()
            input_lnx = model_res.ln_1(x_k)
            ln_x, attn = self.custom_attn(model_res.attn, input_lnx, attn_mask=attn_maps)
            img_features[kth] = ln_x

        # Keep the batch dimension even when batch_size == 1.
        img_features = model_visual.ln_post(img_features)
        img_features_no_proj = img_features.clone()
        if model_visual.proj is not None:
            img_features = img_features @ model_visual.proj

        return img_features, attn, img_features_no_proj

    def modify(self):
        model_transformer = self.model.visual.transformer
        model_visual = self.model.visual

        model_transformer.forward = self.forward_transformer
        model_visual.forward = self.forward_visual

        if self.use_distill:
            model_visual_pri = self.visual_encoder_copy
            model_visual_pri.forward = self.forward_visual_pri

    def classify(self, x: torch.Tensor, text_emb: torch.Tensor):
        x = x / x.norm(dim=-1, keepdim=True)
        norm_text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        logit_per_image = self.logit_scale * x @ norm_text_emb.to(x.dtype).t()
        # TODO: test的时候改similarity patches_similarity = similarity[:, num_of_tokens + 1:, :]

        soft_per_image = logit_per_image.softmax(dim=-1)
        return soft_per_image, logit_per_image

    def _resize(self, image):
        ori_width, ori_height = image.size
        ratio = self.size / min(ori_width, ori_height)
        ori_width, ori_height = ori_width * ratio, ori_height * ratio
        # ori_width, ori_height = 224, 224
        h, w = (int(ori_height / self.patch_size + 0.5) * self.patch_size,
                int(ori_width / self.patch_size + 0.5) * self.patch_size)
        resized_image = image.resize((w, h), Image.BICUBIC)
        return resized_image

    @staticmethod
    def upsample_pos_emb(emb, new_size):
        first, emb = emb[:1, :], emb[1:, :]
        n, d = emb.size(0), emb.size(1)
        size = int(np.sqrt(n))
        emb = emb.permute(1, 0).view(1, d, size, size)
        emb = F.interpolate(emb, size=new_size, mode='bilinear')
        emb = emb.view(d, -1).contiguous().permute(1, 0)
        emb = torch.cat([first, emb], 0)
        return emb.half()

    def classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [template.format(classname) for template in templates]  # format with class
                texts = clip.tokenize(texts).to(self.device)  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights.t()

    def _encode_text_with_templates(self, classnames):
        """
        Args:
            classnames: List[str]
        Returns:
            text_features: Tensor [num_classes, D]
        """
        zeroshot_weights = []

        for classname in classnames:
            texts = [template.format(classname) for template in self.templates]
            texts = clip.tokenize(texts).to(self.device)

            class_embeddings = self.model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()

            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0)  # [C, D]
        return zeroshot_weights

    """
    def encode_text(self, texts, use_template_embedding=False):
        if use_template_embedding:
            if self.mode == "train":
                text_embeddings = self._encode_text_with_templates(texts)
            else:
                with torch.no_grad():
                    text_embeddings = self._encode_text_with_templates(texts)
            return text_embeddings

        # ===== 原始行为（不使用模板） =====
        if self.mode == "train":
            return self.model.encode_text(texts)
        else:
            with torch.no_grad():
                return self.model.encode_text(texts)
    """

    def encode_text(self, texts, use_template_embedding=False, use_prompt=False):
        """
        texts:
            - list[str] (classnames)
            - or tokenized tensor
        """
        if use_prompt and self.prompt_extractor is not None:
            # FIXME：用到prompt_extractor时再改
            return self.prompt_extractor(texts, self.model)

        if use_template_embedding:
            text_features = self._encode_text_with_templates(texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features
        else:  # NOTE: 需要配合训练代码修改，输入的是否为caption_tokenized
            texts = clip.tokenize(texts).to(self.device)
            text_features = self.model.encode_text(texts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features

    """
    def encode_image_pri(self, img: torch.Tensor):
        return self.encode_image_pri(img)
    """

    def encode_image(self, img: torch.Tensor, type: str, select_layers=None):
        """
        visual encoder
        :param img:
        :param type:
        :param select_layers:
        :return: img_feature, attn, sketch_mid_feats
        """
        if type == "sketch":
            img_feature, attn, sketch_mid_feats = self.model.encode_image(img)
            img_feature = img_feature / img_feature.norm(dim=-1, keepdim=True)
            sketch_mid_feats = sketch_mid_feats / sketch_mid_feats.norm(dim=-1, keepdim=True)
            return img_feature, attn, sketch_mid_feats
        elif type == "image":
            img_feature_pri = self.visual_encoder_copy.forward(img, return_layers=select_layers)
            img_feature_pri = img_feature_pri / img_feature_pri.norm(dim=-1, keepdim=True)
            return img_feature_pri

    def forward(self, img: torch.Tensor, tokenized_text=None, text_classes=None, bg_classes=None, pri_img=None, select_layers=None, image_name="", fuse_type="select_layers", visualize_attn=False):
        if self.mode == "train":
            B = img.shape[0]  # 新增：获取批量大小
            self.img_h, self.img_w = img.shape[2] // self.patch_size, img.shape[3] // self.patch_size
            assert isinstance(self.img_h, int) and isinstance(self.img_w,
                                                              int), "Batch images must have the same size"
            if tokenized_text is not None:
                # text_features = self.encode_text(tokenized_text, use_template_embedding=False)  # TODO: 在此处修改是否需要用模板
                text_features = self.encode_text(tokenized_text, use_template_embedding=False, use_prompt=False)
            else:
                # text_features = self.encode_text(text_classes, use_template_embedding=True)
                text_features = self.encode_text(
                    text_classes,
                    use_template_embedding=not self.use_prompt,
                    use_prompt=self.use_prompt
                )

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # 图像特征处理

            # sketch_mid_feats = self.forward_transformer_intermediate(x, return_layers=(7, 10))

            img_feature, attn, sketch_mid_feats = self.model.encode_image(img)

            img_feature = img_feature / img_feature.norm(dim=-1, keepdim=True)
            sketch_mid_feats = sketch_mid_feats / sketch_mid_feats.norm(dim=-1, keepdim=True)

            if self.use_distill and pri_img is not None:
                img_feature_pri = self.visual_encoder_copy.forward(pri_img, return_layers=select_layers)
                img_feature_pri = img_feature_pri / img_feature_pri.norm(dim=-1, keepdim=True)

                return img_feature, text_features, attn, img_feature_pri, sketch_mid_feats

            return img_feature, text_features, attn
        else:  # 即self.mode = “test"
            self.img_name = image_name
            self.img = img
            self.img_h, self.img_w = img.shape[2] // self.patch_size, img.shape[3] // self.patch_size
            # 文本特征处理
            # text_features = torch.cat([fg_text_features, bg_text_features, fg_text_features.mean(0, True)], dim=0)
            # TODO：如何处理bg_text_features
            fg_text_features = self.encode_text(text_classes, use_template_embedding=not self.use_prompt,use_prompt=self.use_prompt)
            bg_text_features = self.encode_text(bg_classes, use_template_embedding=not self.use_prompt,use_prompt=self.use_prompt)
            # redundant_features = self.encode_text([""])
            # text_features = torch.cat([fg_text_features-redundant_features,fg_text_features.mean(0, True)], dim=0)
            text_features = torch.cat([fg_text_features, fg_text_features.mean(0, True)], dim=0)
            # 图像特征处理
            with torch.no_grad():
                img_feature, attn, _ = self.model.encode_image(img)
                logits = self.classify(img_feature, text_features)[1]
                if logits.dim() == 4:
                    # logits: [L, N, B, C]
                    if self.add_prompt:
                        logits = logits[:, 1 + self.num_tokens:, :, :]
                    else:
                        logits = logits[:, 1:, :, :]
                    # [L, N, B, C] -> [L, B, C, N], then merge (L, B)
                    seg = logits.permute(0, 2, 3, 1).contiguous().reshape(
                        -1, len(text_features), self.img_h, self.img_w
                    )
                elif logits.dim() == 3:
                    # logits: [L, N, C]
                    if self.add_prompt:
                        logits = logits[:, 1 + self.num_tokens:, :]
                    else:
                        logits = logits[:, 1:, :]
                    seg = logits.transpose(-1, -2).reshape(-1, len(text_features), self.img_h, self.img_w)
                else:
                    raise ValueError(f"Unexpected logits shape: {logits.shape}")
                seg = seg.softmax(-3)[:, :len(fg_text_features)]

                seg_last = seg[self.select_layer]
                seg_last[seg_last < seg_last.amax((-1, -2), keepdim=True) * self.cfg.attention_thr] = 0
                if self.add_prompt:
                    seg_last = seg_last.flatten(-2, -1) @ attn.mean(0)[1 + self.num_tokens:, 1 + self.num_tokens:]
                else:
                    seg_last = seg_last.flatten(-2, -1) @ attn.mean(0)[1:, 1:]  # 乘以注意力
                seg_last = seg_last.unflatten(dim=-1, sizes=(self.img_h, self.img_w))

                # 可视化注意力矩阵
                """
                if visualize_attn and idx<10:
                    attn_matrix = attn.mean(0)[1:, 1:].detach().cpu().numpy()
                    plt.figure(figsize=(10, 8))
                    plt.imshow(attn_matrix, cmap='viridis')
                    plt.colorbar()
                    plt.title('Attention Matrix at seg_last computation')
                    plt.savefig(f'./attention_map/attention_matrix_{idx}.png')
                    plt.close()
                if idx >= 10:
                    print("查看效果用")
                """
                # if idx==0:
                    # print(fuse_type)
                if fuse_type == "default":
                    seg_to_fuse = seg.mean(0)
                    seg = seg_last + (seg.mean(0) if self.fuse_feature else 0)
                elif fuse_type == "weighted":
                    # 假设有 12 层，我们给最后一层 0.5 的权重，其余 11 层平分剩下的 0.5
                    weight = 0.6
                    num_layers = seg.shape[0]
                    weights = torch.ones(num_layers, device=seg.device) * ((1-weight) / (num_layers - 1))
                    weights[-1] = weight  # 设置最后一层权重
                    # 将 weights 维度对齐到 [Layers, 1, 1, 1] 以便进行广播乘法
                    weights = weights.view(-1, 1, 1, 1)
                    seg_to_fuse = (seg * weights).sum(0)
                elif fuse_type == "select_layers":
                    num_layers_to_fuse = 8
                    seg_to_fuse = seg[-num_layers_to_fuse:].mean(0)
                elif fuse_type == "select_layers + add_weight":
                    weight = 0.5
                    num_layers_to_fuse = 10
                    seg_layers = seg[-num_layers_to_fuse:]
                    num_layers = seg_layers.shape[0]
                    weights = torch.ones(num_layers, device=seg.device) * ((1 - weight) / (num_layers - 1))
                    weights[-1] = weight  # 设置最后一层权重
                    # 将 weights 维度对齐到 [Layers, 1, 1, 1] 以便进行广播乘法
                    weights = weights.view(-1, 1, 1, 1)
                    seg_to_fuse = (seg_layers * weights).sum(0)
                else:
                    seg_to_fuse = 0

                # NOTE：聚合最后N层
                seg = seg_last + seg_to_fuse

            results = {"seg": seg.detach(), "img_part_features": img_feature.clone(), "mid_feature": None,
                       "attn_map": attn.mean(0)[1:, 1:].clone()}

            seg = results["seg"]
            # visualize_attn_map(results["attn_map"], img, patch_size=16)
            # TODO: 把可视化代码移到外面
            if visualize_attn and self.visualize_custom_attn:
                pass

            final_score = seg.amax(dim=(-1, -2))
            torch.cuda.empty_cache()
            return seg, final_score
