import torch
def cohesion_loss(feat_map, gt_mask, feat_mean_stack):
    """intra-mask smoothing loss. Eq.(1) in the paper
    Constrain the feature of each pixel within the mask to be close to the mean feature of that mask.
    """
    N, HW = gt_mask.shape
    C = feat_map.shape[0]
    # expand feat_map [16, HW] to [N, 6, HW]
    feat_map_expanded = feat_map.unsqueeze(0).expand(N, C, HW)
    # expand mean feat [N, 6] to [N, 6, HW]
    feat_mean_stack_expanded = feat_mean_stack.unsqueeze(-1).expand(N, C, HW)
    # print(feat_map_expanded.shape, gt_mask.shape)
    # awdwa
    # fature distance    
    masked_feat = feat_map_expanded * gt_mask.unsqueeze(1)           # [N, 16, HW]
    dist = (masked_feat - feat_mean_stack_expanded).norm(p=2, dim=1) # [N, HW]
    
    # per mask feature distance (loss)
    masked_dist = dist * gt_mask    # [N, HW]
    loss_per_mask = masked_dist.sum(dim=[1]) / gt_mask.sum(dim=[1]).clamp(min=1)

    return loss_per_mask.mean()

def separation_loss(feat_mean_stack, epoch):
    """ inter-mask contrastive loss Eq.(2) in the paper
    Constrain the instance features within different masks to be as far apart as possible.
    """
    N, _ = feat_mean_stack.shape

    # expand feat_mean_stack[N, 6] to [N, N, C]
    feat_expanded = feat_mean_stack.unsqueeze(1).expand(-1, N, -1)
    feat_transposed = feat_mean_stack.unsqueeze(0).expand(N, -1, -1)
    
    # distance
    diff_squared = (feat_expanded - feat_transposed).pow(2).sum(2)
    
    # Calculate the inverse of the distance to enhance discrimination
    epsilon = 1     # 1e-6
    inverse_distance = 1.0 / (diff_squared + epsilon)
    # Exclude diagonal elements (distance from itself) and calculate the mean inverse distance
    mask = torch.eye(N, device=feat_mean_stack.device).bool()
    inverse_distance.masked_fill_(mask, 0)  

    # note: weight
    # sorted by distance
    sorted_indices = inverse_distance.argsort().argsort()
    loss_weight = (sorted_indices.float() / (N - 1)) * (1.0 - 0.1) + 0.1    # scale to 0.1 - 1.0, [N, N]
    # small weight
    if epoch > 10:
        loss_weight[loss_weight < 0.9] = 0.1
    inverse_distance *= loss_weight     # [N, N]

    # final loss
    loss = inverse_distance.sum() / (N * (N - 1))

    return loss

def ele_multip_in_chunks(feat_expanded, masks_expanded, chunk_size=5):
    result = torch.zeros_like(feat_expanded)
    for i in range(0, feat_expanded.size(0), chunk_size):
        end_i = min(i + chunk_size, feat_expanded.size(0))
        for j in range(0, feat_expanded.size(1), chunk_size):
            end_j = min(j + chunk_size, feat_expanded.size(1))
            chunk_feat = feat_expanded[i:end_i, j:end_j]
            chunk_mask = masks_expanded[i:end_i, j:end_j].float()

            result[i:end_i, j:end_j] = chunk_feat * chunk_mask
    return result

def mask_feature_mean(feat_map, gt_masks, image_mask=None, return_var=False):
    """Compute the average instance features within each mask.
    feat_map: [C=6, HW]         the instance features of the entire image
    gt_masks: [num_mask, HW]  num_mask boolean masks
    """
    num_mask, HW = gt_masks.shape

    # expand feat and masks for batch processing
    feat_expanded = feat_map.unsqueeze(0).expand(num_mask, *feat_map.shape)  # [num_mask, C, HW]
    masks_expanded = gt_masks.unsqueeze(1).expand(-1, feat_map.shape[0], -1)  # [num_mask, C, HW]
    if image_mask is not None:  # image level mask
        image_mask_expanded = image_mask.unsqueeze(0).expand(num_mask, feat_map.shape[0], -1)

    # average features within each mask
    if image_mask is not None:
        masked_feats = feat_expanded * masks_expanded.float() * image_mask_expanded.float()
        mask_counts = (masks_expanded * image_mask_expanded.float()).sum(dim=(2))
    else:
        # masked_feats = feat_expanded * masks_expanded.float()  # [num_mask, C, H, W] may cause OOM
        masked_feats = ele_multip_in_chunks(feat_expanded, masks_expanded, chunk_size=20)   # in chuck to avoid OOM
        mask_counts = masks_expanded.sum(dim=(2))  # [num_mask, C]
    # print(masked_feats.shape)
    # awdwa
    # the number of pixels within each mask
    mask_counts = mask_counts.clamp(min=1)

    # the mean features of each mask
    sum_per_channel = masked_feats.sum(dim=[2])
    mean_per_channel = sum_per_channel / mask_counts    # [num_mask, C]

    # print(mean_per_channel.shape)
    # awdaw
    return mean_per_channel   # [num_mask, C]
