import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from configs.config import Config
from dataset import FSCOCOTest
from models.modified_model import ModifiedCLIP
from utils.core import (
    compute_accuracy,
    compute_miou,
    pen_state_to_strokes,
    pixel_level_segmentation,
    prerender_stroke,
)

BICUBIC = InterpolationMode.BICUBIC


def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess = Compose([
        Resize((224, 224), interpolation=BICUBIC),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    dataset = FSCOCOTest(root=cfg.test.root)
    with open(os.path.join(cfg.test.root, "all_classes.json"), "r", encoding="utf-8") as f:
        all_classes = json.load(f)

    model = ModifiedCLIP(cfg, device=device).float().to(device)
    if cfg.checkpoint_path:
        state_dict = torch.load(cfg.checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    pixel_scores, stroke_scores, mious = [], [], []
    total_time = 0.0
    os.makedirs(cfg.test.output_dir, exist_ok=True)

    for pen_state, labels, _, img_path in tqdm(dataset, desc="Evaluating"):
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        strokes = pen_state_to_strokes(pen_state)
        strokes = prerender_stroke(strokes, side=cfg.test.render_size).squeeze(1).float()
        strokes_seg = np.array(strokes)
        sketch = np.array(strokes).sum(0) * 255
        sketch = np.repeat(sketch[np.newaxis, :, :], 3, axis=0)
        sketch = np.transpose(sketch, (1, 2, 0)).astype("uint8")
        sketch = np.where(sketch > 0, 255, sketch)
        sketch = 255 - sketch

        classes = ["blank_pixel"] + np.unique(labels).tolist()
        gt_seg = pixel_level_segmentation(strokes_seg, labels, classes, size=strokes_seg.shape[-1])
        ori_img = torch.tensor(sketch.astype(float)).permute(2, 0, 1) / 255.0
        infer_image = preprocess(Image.fromarray(sketch)).unsqueeze(0).to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            pred_mask, final_score = model(
                infer_image,
                text_classes=classes[1:],
                bg_classes=["blank_pixel"],
                image_name=image_name,
                fuse_type=cfg.fuse_type,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start

        pred_mask = F.interpolate(pred_mask[None], size=ori_img.shape[-2:], mode="bilinear")[0]
        pred_mask[final_score < cfg.class_threshold] = 0
        pred_mask = torch.cat([torch.ones_like(pred_mask[:1]) * cfg.segmentation_threshold, pred_mask])
        pred_seg = pred_mask.argmax(dim=0).cpu().numpy()

        mapping_indices = {i: all_classes.index(j) for i, j in enumerate(classes)}
        pred_seg = np.vectorize(mapping_indices.get)(pred_seg)
        gt_seg = np.vectorize(mapping_indices.get)(gt_seg)
        pred_seg[gt_seg == 0] = 0

        p_acc, s_acc = compute_accuracy(strokes_seg, labels, all_classes, gt_seg, pred_seg)
        pixel_scores.append(p_acc)
        stroke_scores.append(s_acc)
        mious.append(compute_miou(gt_seg, pred_seg, all_classes))

    n = max(len(dataset), 1)
    print(f"Pixel Accuracy: {np.mean(pixel_scores):.5f}")
    print(f"Stroke Accuracy: {np.mean(stroke_scores):.5f}")
    print(f"mIoU: {np.mean(mious):.5f}")
    print(f"Average Forward Time: {total_time / n * 1000:.5f} ms/image")


if __name__ == "__main__":
    config = Config("configs/test_fscoco.yaml")
    config.update_from_cli()
    config.semantic_templates = [line.strip() for line in open(config.semantic_templates, encoding="utf-8")]
    main(config)
