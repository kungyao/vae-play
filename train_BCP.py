import os
import argparse
from datetime import datetime

import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from datasets .dataset import BCPDataset
from models.networks_BCP import ComposeNet, VALUE_WEIGHT
# from test_BP import save_test_batch
from tools.ops import initialize_model
from tools.utils import makedirs

def train_collate_fn(batch):
    imgs, bmasks, labels, annotation = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    bmasks = torch.stack(bmasks, dim=0)
    labels = torch.tensor(labels)
    return imgs, bmasks, labels, annotation

def train(args, epoch, iterations, net, optim, train_loader):
    net.train()
    
    count = 0
    avg_loss = {
        "loss_class": 0, 
        "loss_total_regress": 0, 
        "loss_key_regress": 0, 
    }

    train_iter = iter(train_loader)
    for i in trange(iterations):
        try:
            imgs, bmasks, labels, annotation = next(train_iter)
        except:
            train_iter = iter(train_loader)
            imgs, bmasks, labels, annotation = next(train_iter)

        b = imgs.size(0)
        imgs = imgs.cuda(args.gpu)
        # bmasks = bmasks.cuda(args.gpu)
        labels = labels.cuda(args.gpu)
        annotation = [{k: v.cuda(args.gpu) for k, v in t.items()} for t in annotation]

        preds = net(imgs, target=annotation)

        # size = [len(t["total"]) for t in annotation]

        loss_class = F.cross_entropy(preds["classes"], labels)

        contour_target_gt = torch.cat([t["points"][:, 2:4] for t in annotation], dim=0) * VALUE_WEIGHT
        loss_total_regress = F.l1_loss(
            torch.cat(preds["target_pts"], dim=0), 
            contour_target_gt
        )

        loss_key_regress = []
        for cnt_idx in range(len(annotation)):
            anno = annotation[cnt_idx]
            # 怕有誤差，準確來說是要 == 1.0
            select = anno["points"][:, 5] > 0.9
            if torch.sum(select) != 0:
                loss_key = torch.abs(preds["target_pts"][cnt_idx][select] - anno["points"][select, 2:4] * VALUE_WEIGHT)
                # times length for another weight
                loss_key = loss_key * anno["points"][select, 4]
                loss_key_regress.append(loss_key)
            else:
                loss_key_regress.append(torch.tensor(0.))
        loss_key_regress = torch.mean(torch.stack(loss_key_regress, dim=0))

        losses = loss_class + loss_total_regress + loss_key_regress * 10

        optim.zero_grad()
        losses.backward()
        optim.step()

        with torch.no_grad():
            next_count = count + imgs.size(0)
            avg_loss["loss_class"] = (avg_loss["loss_class"] * count + loss_class.item()) / next_count
            avg_loss["loss_total_regress"] = (avg_loss["loss_total_regress"] * count + loss_total_regress.item()) / next_count
            avg_loss["loss_key_regress"] = (avg_loss["loss_key_regress"] * count + loss_key_regress.item()) / next_count
            count = next_count

            if (i+1) % args.viz_freq == 0:
                print("")
                res_str = f"[Epoch: {epoch}]。"
                for key in avg_loss:
                    res_str += f"{key}: {round(avg_loss[key], 6)}; "
                print(res_str)
                imgs = imgs.cpu()
                _, classes = torch.max(preds["classes"], dim=1)
                split_preds_target_pts = [x.cpu() for x in preds["target_pts"]]
                save_test_batch(imgs, bmasks, classes, preds["contours"], split_preds_target_pts, args.res_output, f"{epoch}_{i+1}")
    return

def save_test_batch(imgs, bmasks, classes, contours, contour_targets, result_path, result_name):
    b, c, h, w = imgs.shape
    
    results = []
    for i in range(b):
        tmp_b = bmasks[i].clone()
        tmp_b = torch.permute(tmp_b, (1, 2, 0))
        tmp_b = np.ascontiguousarray(tmp_b.numpy()).astype(np.uint8) * 255
        # Contour不用 /VALUE_WEIGHT
        cnt = (contours[i] * 0.5 + 0.5) * h
        cnt_target = ((contour_targets[i] / VALUE_WEIGHT) * 0.5 + 0.5) * h
        cnt_size = len(cnt)
        if classes[i] == 1:
            for j in range(cnt_size):
                pt = cnt[j].to(dtype=torch.long).tolist()
                end_pt = cnt_target[j].to(dtype=torch.long).tolist()
                cv2.line(tmp_b, (pt[0], pt[1]), (end_pt[0], end_pt[1]), (255, 255, 255), thickness=1)
        else:
            for j in range(cnt_size):
                pt = cnt_target[j].to(dtype=torch.long).tolist()
                end_pt = cnt_target[(j+1)%cnt_size].to(dtype=torch.long).tolist()
                cv2.line(tmp_b, (pt[0], pt[1]), (end_pt[0], end_pt[1]), (255, 255, 255), thickness=1)
        tmp_b = TF.to_tensor(tmp_b)
        results.append(tmp_b)
    results = torch.stack(results, dim=0)
    vutils.save_image(
        torch.cat([imgs, bmasks, results], dim=0), 
        os.path.join(result_path, f"{result_name}.png"),
        nrow=b, 
        padding=2, 
        pad_value=1
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', type=str, dest='path', default="D:/Manga/bubble-cnt-data")
    # 
    parser.add_argument('--lr', type=float, dest='lr', default=1e-3)
    parser.add_argument('--gpu', type=int, dest='gpu', default=0)
    parser.add_argument('--epoch', type=int, dest='epochs', default=1)
    parser.add_argument('--iterations', type=int, dest='iterations', default=200)
    parser.add_argument('--batchsize', type=int, dest='batchsize', default=16)
    #
    parser.add_argument('--workers', type=int, dest='workers', default=0)
    # 
    parser.add_argument('--img_size', type=int, dest='img_size', default=512)
    # parser.add_argument('--max_points', type=int, dest='max_points', default=DEFAULT_MAX_POINTS)
    #
    parser.add_argument('--res_output', type=str, dest='res_output', default='./results')
    parser.add_argument('--model_output', type=str, dest='model_output', default='./logs')
    parser.add_argument('--viz_freq', type=int, dest='viz_freq', default=10)
    args = parser.parse_args()

    dest_name = os.path.join("BCP", datetime.now().strftime("%Y%m%d-%H%M%S"))
    args.res_output = os.path.join(args.res_output, dest_name)
    args.model_output = os.path.join(args.model_output, dest_name)

    makedirs(args.res_output)
    makedirs(args.model_output)

    record_txt = open(os.path.join(args.model_output, "record.txt"), "w")
    for arg in vars(args):
        record_txt.write('{:35}{:20}\n'.format(arg, str(getattr(args, arg))))
    record_txt.close()

    # width height
    dset = BCPDataset(args.path, args.img_size)
    dloader = DataLoader(
        dset, 
        batch_size=args.batchsize, 
        shuffle=True, 
        num_workers=args.workers, 
        collate_fn=train_collate_fn, 
        pin_memory=True)
    
    net = ComposeNet(args.img_size)
    initialize_model(net.encoder)
    initialize_model(net.line_predictor)

    net.cuda(args.gpu)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    step_size = 2
    # step_size = args.epochs // 4
    step_size = 1 if step_size == 0 else step_size
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size, gamma=0.1)

    for epoch in range(args.epochs):
        train(args, epoch, args.iterations, net, optim, dloader)
        torch.save(
            {
                "networks": net, 
                # "optims": optim,
                "epoch": epoch
            }, 
            os.path.join(args.model_output, f"{epoch}.ckpt")
        )
        scheduler.step()

    # for i, (imgs, bmasks, labels, annotations) in enumerate(dloader):
    #     b, c, h, w = imgs.shape
    #     contours = []
    #     contour_targets = []
    #     for x in annotations:
    #         print(len(x["total"]))
    #         contours.append(x["total"][:, :2])
    #         contour_targets.append(x["total"][:, 2:4] * VALUE_WEIGHT)
    #     save_test_batch(imgs, bmasks, labels, contours, contour_targets, result_path="./results", result_name=f"test_{i}")
