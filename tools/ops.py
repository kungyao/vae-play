import torch
import torch.nn.functional as F

def compute_dice_loss(inputs, targets, smooth = 1.):
    nums = inputs.size(0)
    iflat = inputs.view(nums, -1)
    tflat = targets.view(nums, -1)
    intersection = iflat * tflat
    score = (2. * intersection.sum(1) + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth)
    score = 1 - score.sum() / nums
    return score

def compute_pt_regression_loss(predict_contours, predict_regressions, target_contours):
    """
    Args:
        predict_contours ([torch.Tensor]): B*N*2 (0<N<=MAX_POINTS, N can be different in each batch.)
        predict_regressions (torch.Tensor): B*MAX_POINTS*2
        target_contours ([torch.Tensor]): B*M*2 (0<M<=MAX_INT, M can be different in each batch.)
    """
    # Find minimum distance between the contours and return regressions.
    b = len(predict_contours)
    losses = []
    for i in range(b):
        pcnt = predict_contours[i]
        N = pcnt.size(0)
        p_regress = predict_regressions[i].cpu()
        device = p_regress.device
        if N != 0:
            tcnt = target_contours[i]
            M = tcnt.size(0)
            # N*M*2
            # "tcnt - pcnt" because we also need to know the regression vector.
            dif_mat = tcnt.unsqueeze(0).repeat(N, 1, 1) - pcnt.unsqueeze(1).repeat(1, M, 1)
            # N*M
            distance_mat = torch.norm(dif_mat, dim=-1)
            # We don't need the value
            # N, Find closest points in "target_contours" using "predict_contours".
            _, p2t_idx = torch.min(distance_mat, dim=1)
            # M, Find closest points in "predict_contours" using "target_contours".
            _, t2p_idx = torch.min(distance_mat, dim=0)
            # .to(device)
            # N
            loss_p2t = F.mse_loss(p_regress[torch.arange(N)], dif_mat[torch.arange(N), p2t_idx], reduction='mean')
            # M
            loss_t2p = F.mse_loss(p_regress[t2p_idx], dif_mat[t2p_idx, torch.arange(M)], reduction='mean')
            losses.append(loss_p2t + loss_t2p)
        else:
            losses.append(p_regress.sum() * 0)
    losses = torch.sum(torch.stack(losses))
    return losses

def compute_hinge_loss(logit, mode):
    assert mode in ['d_real', 'd_fake', 'g']
    if mode == 'd_real':
        loss = F.relu(1.0 - logit).mean()
    elif mode == 'd_fake':
        loss = F.relu(1.0 + logit).mean()
    else:
        loss = -logit.mean()
    return loss

