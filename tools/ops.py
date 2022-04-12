import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.utils import angle_between

VALUE_WEIGHT = 10

def compute_dice_loss(inputs, targets, smooth = 1.):
    nums = inputs.size(0)
    iflat = inputs.view(nums, -1)
    tflat = targets.view(nums, -1)
    intersection = iflat * tflat
    score = (2. * intersection.sum(1) + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth)
    score = 1 - score.sum() / nums
    return score

def compute_pt_regression_loss(predict_contours, predict_regressions, target_contours, target_key_contours):
    """
    Args:
        predict_contours ([torch.Tensor]): B*N*2 (0<N<=MAX_POINTS, N can be different in each batch.)
        predict_regressions (torch.Tensor): B*MAX_POINTS*2
        target_contours ([torch.Tensor]): B*M*2 (0<M<=MAX_INT, M can be different in each batch.)
    """
    def sub_contour_loss(predict_cnt, predict_regress, target_cnt, weight_p2t:float =1.0, weight_t2p:float =1.0):
        N = predict_cnt.size(0)
        M = target_cnt.size(0)           
         # N*M*2
        # "target_cnt - predict_cnt" because we also need to know the regression vector.
        dif_mat = target_cnt.unsqueeze(0).repeat(N, 1, 1) - predict_cnt.unsqueeze(1).repeat(1, M, 1)
        # N*M
        distance_mat = torch.norm(dif_mat, dim=-1)
        # We don't need the value
        # N, Find closest points in "target_contours" using "predict_contours".
        _, p2t_idx = torch.min(distance_mat, dim=1)
        # M, Find closest points in "predict_contours" using "target_contours".
        _, t2p_idx = torch.min(distance_mat, dim=0)
        # .to(device)
        # N
        loss_p2t = F.mse_loss(predict_regress[torch.arange(N)], dif_mat[torch.arange(N), p2t_idx], reduction='mean')
        # M
        loss_t2p = F.mse_loss(predict_regress[t2p_idx], dif_mat[t2p_idx, torch.arange(M)], reduction='mean')
        return weight_p2t*loss_p2t + weight_t2p*loss_t2p
    
    # Find minimum distance between the contours and return regressions.
    b = len(predict_contours)
    losses = []
    for i in range(b):
        p_cnt = predict_contours[i]
        N = p_cnt.size(0)
        p_regress = predict_regressions[i].cpu()
        device = p_regress.device
        if N != 0:
            # 讓predict point貼合ground truth。
            loss = sub_contour_loss(p_cnt, p_regress, target_contours[i], weight_p2t=1.0, weight_t2p=0.1)
            # 計算原始contour的特異點(RDP)，然後利用這組新的contour計算一個新的loss，不然利用全部點去算loss_t2p的話，總loss會被平均掉。
            # 設weight_p2t為0，著重在每個key point是否有被找到。
            loss_key = sub_contour_loss(p_cnt, p_regress, target_key_contours[i], weight_p2t=0.0, weight_t2p=2.0)
            losses.append(loss+loss_key)
        else:
            losses.append(p_regress.sum() * 0)
    losses = torch.mean(torch.stack(losses))
    return losses

def compute_ellipse_param_loss(preds, gt_targets):
    gt_targets = gt_targets.to(preds.device)
    # Weight
    gt_targets[:, :4] = gt_targets[:, :4] * VALUE_WEIGHT
    # loss = F.l1_loss(preds, gt_targets)
    loss_cx = F.l1_loss(preds[:, 0], gt_targets[:, 0]) * 2
    loss_cy = F.l1_loss(preds[:, 1], gt_targets[:, 1]) * 2
    loss_rest = F.l1_loss(preds[:, 2:], gt_targets[:, 2:])
    # print(preds, gt_targets)
    return {
        "loss_cx": loss_cx, 
        "loss_cy": loss_cy, 
        "loss_rest": loss_rest
    }

def compute_ellipse_pt_loss(preds, gt_targets):
    pred_triggers = preds["if_triggers"]
    # offset_x, offset_y, theta, length
    pred_line_params = preds["line_params"]
    # Known by predicted ellipse params
    pred_sample_size = preds["sample_infos"]["size"]
    # px, py, dpx, dpy, d
    pred_sample_sample = preds["sample_infos"]["sample"]

    # Collect matches
    loss_target_trig = []
    loss_target_param = []
    for i in range(len(gt_targets)):
        target = gt_targets[i]
        # p_triggers = pred_triggers[i]
        # p_line_params = pred_line_params[i]
        p_sample_sample = pred_sample_sample[i]
        p_sample_dense = p_sample_sample[:, 4].to(dtype=torch.long)

        ts = target[p_sample_dense]
        new_target_trig = ts[:, 0]
        # new_target_param = torch.stack([
        #     (ts[:, 1] - p_sample_sample[:, 0]) * VALUE_WEIGHT, 
        #     (ts[:, 2] - p_sample_sample[:, 1]) * VALUE_WEIGHT, 
        #     (ts[:, 1] + ts[:, 3] * ts[:, 5]) * VALUE_WEIGHT, 
        #     (ts[:, 2] + ts[:, 4] * ts[:, 5]) * VALUE_WEIGHT, 
        # ], dim=-1)
        new_target_param = torch.stack([
            (ts[:, 1] - p_sample_sample[:, 0]) * VALUE_WEIGHT, 
            (ts[:, 2] - p_sample_sample[:, 1]) * VALUE_WEIGHT, 
            torch.arccos(torch.clip(ts[:, 3] * p_sample_sample[:, 2] + ts[:, 4] * p_sample_sample[:, 3], -1.0, 1.0)), 
            (ts[:, 5] * VALUE_WEIGHT)
        ], dim=-1)
        # new_target_param = ts[:, 5] * VALUE_WEIGHT
        # new_target_param = new_target_param.reshape(-1, 1)

        loss_target_trig.append(torch.FloatTensor(new_target_trig))
        loss_target_param.append(torch.FloatTensor(new_target_param))
        
    # 
    pred_triggers = torch.cat(pred_triggers, dim=0)
    pred_line_params = torch.cat(pred_line_params, dim=0)
    # 
    # loss_target_trig = torch.cat(loss_target_trig, dim=0).to(pred_triggers.device)
    loss_target_trig = torch.cat(loss_target_trig, dim=0).to(dtype=torch.long, device=pred_triggers.device)
    loss_target_param = torch.cat(loss_target_param, dim=0).to(pred_triggers.device)
    # 
    trig_idx = loss_target_trig >= 0.5
    non_trig_idx = loss_target_trig < 0.5
    # # Do data balance
    # trig_sum = torch.sum(trig_idx)
    # non_trig_sum = torch.sum(non_trig_idx)
    # 
    # Resample data by under-sample mode.
    # if trig_sum > non_trig_sum:
    #     trig_select = torch.randperm(trig_sum)[:non_trig_sum]
    #     pred_line_params = torch.cat([pred_line_params[trig_select], pred_line_params[non_trig_idx]], dim=0)
    #     loss_target_param = torch.cat([loss_target_param[trig_select], loss_target_param[non_trig_idx]], dim=0)
    # elif non_trig_sum > trig_sum:
    #     non_trig_select = torch.randperm(non_trig_sum)[:trig_sum]
    #     pred_line_params = torch.cat([pred_line_params[trig_idx], pred_line_params[non_trig_select]], dim=0)
    #     loss_target_param = torch.cat([loss_target_param[trig_idx], loss_target_param[non_trig_select]], dim=0)
    # 
    # trig_loss = F.binary_cross_entropy(pred_triggers.squeeze(), loss_target_trig)
    # trig_loss = F.cross_entropy(pred_triggers, loss_target_trig)
    trig_loss = F.cross_entropy(pred_triggers[trig_idx], loss_target_trig[trig_idx], reduction='mean') + F.cross_entropy(pred_triggers[non_trig_idx], loss_target_trig[non_trig_idx], reduction='mean')
    # 
    pred_triggers = F.softmax(pred_triggers, dim=-1)
    trig_loss = trig_loss + (compute_dice_loss(pred_triggers[:, 0], 1 - loss_target_trig) + compute_dice_loss(pred_triggers[:, 1], loss_target_trig))/2
    # trig_loss = trig_loss + compute_dice_loss(pred_triggers[:, 1], loss_target_trig) * 2
    trig_loss = trig_loss * 2
    # param_loss = F.l1_loss(pred_line_params[trig_idx], loss_target_param[trig_idx])
    # param_loss = F.l1_loss(pred_line_params[trig_idx], loss_target_param[trig_idx], reduction='mean') + F.l1_loss(pred_line_params[non_trig_idx], loss_target_param[non_trig_idx], reduction='mean')
    param_normal_loss = F.l1_loss(pred_line_params[trig_idx][:, :3], loss_target_param[trig_idx][:, :3], reduction='mean') + F.l1_loss(pred_line_params[non_trig_idx][:, :3], loss_target_param[non_trig_idx][:, :3], reduction='mean')
    # param_length_loss = torch.sqrt(torch.square(pred_line_params[trig_idx][:, 3] - loss_target_param[trig_idx][:, 3]))
    # param_length_loss = torch.sum(param_length_loss / (torch.sum(param_length_loss > 1e-2) + 1))
    # param_length_loss = param_length_loss
    param_length_loss = F.mse_loss(pred_line_params[trig_idx][:, 3], loss_target_param[trig_idx][:, 3], reduction='mean') + F.l1_loss(pred_line_params[trig_idx][:, 3], loss_target_param[trig_idx][:, 3], reduction='mean')
    param_loss = param_length_loss + param_normal_loss
    # loss = trig_loss + param_loss
    return {
        "trig_loss":  trig_loss,
        "param_loss":  param_loss
    }

def compute_hinge_loss(logit, mode):
    assert mode in ['d_real', 'd_fake', 'g']
    if mode == 'd_real':
        loss = F.relu(1.0 - logit).mean()
    elif mode == 'd_fake':
        loss = F.relu(1.0 + logit).mean()
    else:
        loss = -logit.mean()
    return loss

def initialize_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

