import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from configs.config import Config
from dataset import FSCOCOTrain
from models.modified_model import ModifiedCLIP
from utils.core import (
    get_attention_map,
    get_threshold,
    get_train_classes,
    load_checkpoint,
    save_checkpoint,
    sketch_text_pairs,
    tensor_to_binary_img,
    zero_clapping_attn,
)
from utils.losses import triplet_loss_func_l1

BICUBIC = InterpolationMode.BICUBIC


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_no_tensor = Compose([
        Resize((224, 224), interpolation=BICUBIC),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    preprocess = Compose([
        Resize((224, 224), interpolation=BICUBIC),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    model = ModifiedCLIP(cfg=cfg, device=device).float().to(device)
    train_dataset = FSCOCOTrain(root=cfg.dataset.root, transform=preprocess, augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )

    learnable_threshold = nn.Parameter(torch.tensor(cfg.train.threshold, device=device))
    threshold_optimizer = torch.optim.AdamW([learnable_threshold], lr=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))

    ckpt_dir = os.path.join(cfg.output_dir, cfg.wandb.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    train_classes_path = os.path.join(ckpt_dir, "train_classes.json")
    last_ckpt_path = os.path.join(ckpt_dir, "last.pth")

    start_epoch = 0
    global_step = 0
    train_classes = None
    if cfg.train.resume and os.path.exists(last_ckpt_path):
        start_epoch, global_step, train_classes = load_checkpoint(
            last_ckpt_path, model, optimizer, threshold_optimizer, device
        )

    if train_classes is None:
        if os.path.exists(train_classes_path):
            train_classes = np.array(json.load(open(train_classes_path, "r", encoding="utf-8")))
        else:
            train_classes = get_train_classes(train_dataset, max_classes=cfg.train.max_classes)
            with open(train_classes_path, "w", encoding="utf-8") as f:
                json.dump(train_classes.tolist(), f, ensure_ascii=False, indent=2)

    model.train()
    for epoch in range(start_epoch, cfg.train.epochs):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")
        for sketches, captions in pbar:
            sketches = sketches.view(-1, 3, 224, 224).to(device)
            sketches_w, classes, captions_pair = sketch_text_pairs(
                sketches, captions, max_classes=cfg.train.max_classes
            )
            sketches_binary = tensor_to_binary_img(sketches_w, device)
            sketches_bg = 1 - sketches_binary

            caption_features = model.encode_text(captions_pair)
            class_features = model.encode_text(classes, use_template_embedding=True)
            scene_features_layers, attn, _ = model.encode_image(sketches_w, type="sketch")
            scene_features = scene_features_layers[-1].permute(1, 0, 2)

            attn_map = get_attention_map(attn, image_size=sketches_w.shape[2:], patch_size=model.patch_size)
            threshold_value = get_threshold(learnable_threshold)
            weights = zero_clapping_attn(attn_map, threshold_value).unsqueeze(1).repeat(1, 3, 1, 1)
            weighted_sketches = sketches_bg * weights
            weighted_sketches = weighted_sketches.max() - weighted_sketches
            weighted_sketches = preprocess_no_tensor(weighted_sketches)

            weighted_features, _, _ = model.encode_image(weighted_sketches, type="sketch")
            class_to_idx = {name: i for i, name in enumerate(train_classes)}
            labels = torch.tensor([class_to_idx[name] for name in classes], device=device)

            loss = triplet_loss_func_l1(scene_features[:, 0, :], caption_features, labels, cfg.train.margin)
            for layer_idx in (7, 10, 12):
                layer_features = weighted_features[layer_idx - 1].permute(1, 0, 2)
                loss = loss + triplet_loss_func_l1(layer_features[:, 0, :], class_features, labels, cfg.train.margin)

            optimizer.zero_grad()
            threshold_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            threshold_optimizer.step()

            global_step += 1
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", threshold=f"{threshold_value.item():.4f}")

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "threshold_optimizer": threshold_optimizer.state_dict(),
            "train_classes": train_classes,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
        }
        save_checkpoint(checkpoint, last_ckpt_path)
        if epoch == 0 or (epoch + 1) % int(cfg.train.save_every) == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_{epoch + 1}.pth"))
        print(f"Epoch {epoch + 1}: avg loss = {total_loss / max(len(train_loader), 1):.4f}")


if __name__ == "__main__":
    config = Config("configs/train_fscoco.yaml")
    config.update_from_cli()
    config.semantic_templates = [line.strip() for line in open(config.semantic_templates, encoding="utf-8")]
    main(config)
