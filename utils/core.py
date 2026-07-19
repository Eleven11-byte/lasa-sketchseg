import json
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F


def extract_classes_from_caption(caption, max_classes=3):
    text = caption.strip().lower()
    for prefix in ("a scene sketch containing", "a sketch containing", "containing"):
        text = text.replace(prefix, "")
    text = re.sub(r"[.;:]", ",", text)
    text = text.replace(" and ", ",")
    classes = [c.strip(" ,.") for c in text.split(",") if c.strip(" ,.")]
    return list(dict.fromkeys(classes))[:max_classes]


def sketch_text_pairs(sketch_batch, captions, max_classes=3):
    sketches = []
    classes = []
    caption_pairs = []
    for sketch, caption in zip(sketch_batch, captions):
        cls = extract_classes_from_caption(caption, max_classes=max_classes)
        if not cls:
            cls = ["object"]
        sketches.append(sketch.repeat(len(cls), 1, 1, 1))
        classes.extend(cls)
        caption_pairs.extend([caption] * len(cls))
    return torch.cat(sketches), classes, caption_pairs


def get_train_classes(dataset, max_classes=3):
    all_classes = []
    for _, caption in dataset:
        all_classes.extend(extract_classes_from_caption(caption, max_classes=max_classes))
    return np.array(sorted(set(all_classes)))


def tensor_to_binary_img(tensor, device):
    gray = tensor.mean(dim=1, keepdim=True)
    return (gray < gray.mean(dim=(-1, -2), keepdim=True)).float().to(device)


def zero_clapping_attn(attn_map, threshold):
    return (attn_map >= threshold).float()


def get_threshold(learnable_threshold):
    return torch.sigmoid(learnable_threshold)


def get_attention_map(attn, image_size, patch_size, use_cls=True):
    if attn.dim() == 3:
        attn = attn.mean(0)
    if use_cls:
        attn = attn[0, 1:]
    else:
        attn = attn[1:, 1:].mean(0)
    h = image_size[-2] // patch_size
    w = image_size[-1] // patch_size
    attn = attn.reshape(-1, 1, h, w)
    attn = F.interpolate(attn, size=image_size[-2:], mode="bilinear", align_corners=False)
    attn = attn.squeeze(1)
    attn_min = attn.amin(dim=(-1, -2), keepdim=True)
    attn_max = attn.amax(dim=(-1, -2), keepdim=True)
    return (attn - attn_min) / (attn_max - attn_min + 1e-6)


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, threshold_optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    threshold_optimizer.load_state_dict(ckpt["threshold_optimizer"])
    rng = ckpt.get("rng_state", {})
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and "cuda" in rng:
        torch.cuda.set_rng_state_all(rng["cuda"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "random" in rng:
        random.setstate(rng["random"])
    return ckpt.get("epoch", 0) + 1, ckpt.get("global_step", 0), ckpt.get("train_classes", None)


def pen_state_to_strokes(sketches):
    strokes = []
    start = 0
    for i in range(len(sketches)):
        if sketches[i, 2] == 1:
            strokes.append(sketches[start:i + 1])
            start = i + 1
    return strokes


def _draw_line(canvas, x0, y0, x1, y1):
    x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < canvas.shape[1] and 0 <= y0 < canvas.shape[0]:
            canvas[y0, x0] = 1.0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def prerender_stroke(stroke_list, side=512):
    rendered = []
    for stroke in stroke_list:
        points = np.asarray(stroke, dtype=float).copy()
        points[:, :2] = np.round(points[:, :2] / 256.0 * side)
        canvas = np.zeros((side, side), dtype=np.float32)
        x_prev, y_prev = points[0, :2]
        for i, point in enumerate(points):
            if i > 0 and points[i - 1, 2] == 1:
                x_prev, y_prev = point[:2]
            _draw_line(canvas, x_prev, y_prev, point[0], point[1])
            x_prev, y_prev = point[:2]
        rendered.append(torch.from_numpy(canvas).unsqueeze(0))
    return torch.stack(rendered, 0)


def pixel_level_segmentation(strokes_seg, labels, all_classes, size):
    seg = np.zeros((len(all_classes), size, size), dtype=np.float32)
    blank = all_classes.index("blank_pixel")
    seg[blank] = 1.0
    for stroke, label in zip(strokes_seg, labels):
        class_idx = all_classes.index(label)
        seg[class_idx] += stroke
        seg[blank] -= stroke
    return np.argmax(seg, axis=0)


def compute_accuracy(strokes_seg, labels, classes, gt_seg, pred_seg):
    blank = classes.index("blank_pixel")
    mask = gt_seg != blank
    pixel_acc = float((gt_seg[mask] == pred_seg[mask]).mean()) if mask.any() else 0.0
    label_indices = [classes.index(label) for label in labels]
    pred_strokes = []
    for stroke in strokes_seg:
        values = pred_seg[stroke == 1]
        if len(values) == 0:
            pred_strokes.append(blank)
        else:
            pred_strokes.append(int(np.bincount(values.astype(int)).argmax()))
    stroke_acc = float(np.mean(np.array(label_indices) == np.array(pred_strokes))) if label_indices else 0.0
    return pixel_acc, stroke_acc


def compute_miou(gt_seg, pred_seg, all_classes):
    ious = []
    for c in range(1, len(all_classes)):
        union = np.logical_or(gt_seg == c, pred_seg == c).sum()
        if union:
            ious.append(np.logical_and(gt_seg == c, pred_seg == c).sum() / union)
    return float(np.mean(ious)) if ious else 0.0


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
