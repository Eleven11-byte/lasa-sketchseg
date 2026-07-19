import torch
import torch.nn.functional as F


def triplet_loss_func_l1(sketch_embeddings, class_embeddings, labels, margin=0.2):
    distances = torch.cdist(sketch_embeddings.float(), class_embeddings.float(), p=1)
    if distances.shape[0] != distances.shape[1]:
        raise ValueError(
            "triplet_loss_func_l1 expects paired sketch/text features with the same batch size, "
            f"got {distances.shape[0]} and {distances.shape[1]}."
        )
    batch_size = distances.shape[0]
    diag_mask = torch.eye(batch_size, dtype=torch.bool, device=sketch_embeddings.device)
    positive = distances[diag_mask]
    class_mask = labels[:, None] == labels[None, :]
    negative = distances.masked_fill(diag_mask | class_mask, float("inf")).min(dim=1).values
    valid_negative = torch.isfinite(negative)
    if not valid_negative.any():
        return positive.sum() * 0.0
    return F.relu(positive[valid_negative] - negative[valid_negative] + margin).mean()
