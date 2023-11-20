import torch
from torch import nn
import numpy as np
from sklearn import metrics
import copy
import cv2
import torchvision
from tqdm import tqdm
import os
from torch.utils.data import Dataset
from hausdorff import hausdorff_distance


class Lung_cons(Dataset):
    def __init__(self, csv_file, img_dir, lung_dir, mask_dir):
        self.annotations = csv_file
        self.img_dir = img_dir
        self.lung_dir = lung_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # grab the image path from the current index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image_fused = np.zeros((3, 224, 224))
        image_fused[0] = image / 255
        image_fused[1] = image / 255
        image_fused[2] = image / 255
        image_fused = torch.from_numpy(image_fused)

        if self.annotations.iloc[index, 1] == 1:
            lung_path = os.path.join(self.lung_dir, self.annotations.iloc[index, 0])
            lung = cv2.imread(lung_path, cv2.IMREAD_GRAYSCALE) / 255
        else:
            lung = np.ones((224, 224))

        lung = torch.from_numpy(lung)
        mask_path_tmp = os.path.join(self.mask_dir, self.annotations.iloc[index, 0])
        mask = cv2.imread(mask_path_tmp, cv2.IMREAD_GRAYSCALE) / 255

        return image_fused, lung, mask


class Lung_cons_ext(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir):
        self.annotations = csv_file
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # grab the image path from the current index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        image_fused = np.zeros((3, 224, 224))
        image_fused[0] = image / 255
        image_fused[1] = image / 255
        image_fused[2] = image / 255
        image_fused = torch.from_numpy(image_fused)

        mask_path_tmp = os.path.join(self.mask_dir, self.annotations.iloc[index, 0])

        mask = cv2.imread(mask_path_tmp, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224, 224)) / 255
        mask[np.where(mask != 0)] = 1

        return image_fused, mask


class Lung_cons_tem(Dataset):
    def __init__(self, csv_file, img_dir, lung_dir, mask_dir, tem):
        self.annotations = csv_file
        self.img_dir = img_dir
        self.lung_dir = lung_dir
        self.mask_dir = mask_dir
        self.tem = tem

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # grab the image path from the current index
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image_fused = np.zeros((3, 224, 224))
        image_fused[0] = image / 255
        image_fused[1] = image / 255
        image_fused[2] = image / 255
        image_fused = torch.from_numpy(image_fused)

        if self.annotations.iloc[index, 1] == 1:
            lung_path = os.path.join(self.lung_dir, self.annotations.iloc[index, 0])
            lung = cv2.imread(lung_path, cv2.IMREAD_GRAYSCALE) / 255
            lung[np.where(lung == 0)] = self.tem
        else:
            lung = np.ones((224, 224))

        lung = torch.from_numpy(lung)
        mask_path_tmp = os.path.join(self.mask_dir, self.annotations.iloc[index, 0])
        mask = cv2.imread(mask_path_tmp, cv2.IMREAD_GRAYSCALE) / 255

        return image_fused, lung, mask


class DiceLossCons(nn.Module):
    def __init__(self):
        super(DiceLossCons, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target, lung, lam):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = predict.view(num, -1)
        tar = target.view(num, -1)
        lun = lung.view(num, -1)

        intersection = (pre * tar).sum(-1)
        union = (pre + tar).sum(-1)

        lun_inter = (pre * lun).sum(-1)
        pre_size = pre.sum(-1)

        score = 1 - (2 * (intersection + self.epsilon) / (union + self.epsilon)).mean() + \
            lam * (1 - (lun_inter / pre_size).mean())

        return score


def train_cons(train_loader_seg, device, seg_model, criterion, optimizer, lam):
    seg_model.train()
    losses = 0
    for sample, lung_dilation, targets in tqdm(train_loader_seg):
        sample = sample.type(torch.FloatTensor).to(device=device)
        targets = targets.to(device=device)
        lung_dilation = lung_dilation.type(torch.FloatTensor).to(device=device)

        scores = seg_model(sample)
        scores = torch.sigmoid(scores).squeeze().float()

        loss = criterion(scores, targets.float().squeeze(), lung_dilation, lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss

    return losses / len(train_loader_seg)


def valid_cons(val_loader_seg, device, seg_model, criterion, lam):
    with torch.no_grad():
        losses = 0
        for sample, lung_dilation, targets in tqdm(val_loader_seg):
            sample = sample.type(torch.FloatTensor).to(device=device)
            targets = targets.to(device=device)
            lung_dilation = lung_dilation.type(torch.FloatTensor).to(device=device)

            scores = seg_model(sample)
            scores = torch.sigmoid(scores).squeeze()

            loss = criterion(scores, targets.float().squeeze(), lung_dilation, lam)
            losses += loss

    return losses / len(val_loader_seg)


def test_cons(test_loader_seg, device, seg_model):
    with torch.no_grad():
        iou_list_test_case = []
        dice_list_test_case = []
        hit_list_test_case = []
        hau_list_test_case = []

        for sample, _, targets in tqdm(test_loader_seg):
            sample = sample.type(torch.FloatTensor).to(device)
            targets = np.array(targets.detach().cpu()).flatten()
            outputs = seg_model(sample)

            if targets[np.argmax(outputs.squeeze().detach().cpu().numpy().flatten())] == 1:
                hit_list_test_case.append(1)
            else:
                hit_list_test_case.append(0)

            outputs = torch.sigmoid(outputs).squeeze().detach().cpu().numpy().flatten()

            binary_list = copy.deepcopy(outputs)
            binary_list[np.where(binary_list >= 0.5)] = 1
            binary_list[np.where(binary_list != 1)] = 0
            binary_list = binary_list.astype("uint8")

            iou = metrics.jaccard_score(targets, binary_list)
            iou = format(iou, '.3f')
            dice = metrics.f1_score(targets, binary_list)
            dice = format(dice, '.3f')
            iou_list_test_case.append(iou)
            dice_list_test_case.append(dice)

            hau = hausdorff_distance(np.resize(binary_list, (224, 224)), np.resize(targets, (224, 224)))
            hau = format(hau, '.3f')
            hau_list_test_case.append(hau)

        iou_case = format(np.array(iou_list_test_case).astype('float').mean(), '.3f')
        dice_case = format(np.array(dice_list_test_case).astype('float').mean(), '.3f')
        hit_case = format(np.array(hit_list_test_case).astype('float').mean(), '.3f')
        hau_case = format(np.array(hau_list_test_case).astype('float').mean(), '.3f')

        print(f'IoU on sick test set is {iou_case}')
        print(f'Dice on sick test set is {dice_case}')
        print(f'Hit rate on sick test set is {hit_case}')
        print(f'Hausdorff distance on sick test set is {hau_case}')

        return iou_list_test_case, dice_list_test_case, hit_list_test_case, hau_list_test_case, \
               iou_case, dice_case, hit_case, hau_case


def test_cons_ext(test_loader_seg, device, seg_model):
    with torch.no_grad():
        iou_list_test_case = []
        dice_list_test_case = []
        hit_list_test_case = []
        hau_list_test_case = []

        for sample, targets in tqdm(test_loader_seg):
            sample = sample.type(torch.FloatTensor).to(device)
            targets = np.array(targets.detach().cpu()).flatten()
            outputs = seg_model(sample)

            if targets[np.argmax(outputs.squeeze().detach().cpu().numpy().flatten())] == 1:
                hit_list_test_case.append(1)
            else:
                hit_list_test_case.append(0)

            outputs = torch.sigmoid(outputs).squeeze().detach().cpu().numpy().flatten()

            binary_list = copy.deepcopy(outputs)
            binary_list[np.where(binary_list >= 0.5)] = 1
            binary_list[np.where(binary_list != 1)] = 0
            binary_list = binary_list.astype("uint8")

            iou = metrics.jaccard_score(targets, binary_list)
            iou = format(iou, '.3f')
            dice = metrics.f1_score(targets, binary_list)
            dice = format(dice, '.3f')
            iou_list_test_case.append(iou)
            dice_list_test_case.append(dice)

            hau = hausdorff_distance(np.resize(binary_list, (224, 224)), np.resize(targets, (224, 224)))
            hau = format(hau, '.3f')
            hau_list_test_case.append(hau)

        iou_case = format(np.array(iou_list_test_case).astype('float').mean(), '.3f')
        dice_case = format(np.array(dice_list_test_case).astype('float').mean(), '.3f')
        hit_case = format(np.array(hit_list_test_case).astype('float').mean(), '.3f')
        hau_case = format(np.array(hau_list_test_case).astype('float').mean(), '.3f')

        print(f'IoU on sick test set is {iou_case}')
        print(f'Dice on sick test set is {dice_case}')
        print(f'Hit rate on sick test set is {hit_case}')
        print(f'Hausdorff distance on sick test set is {hau_case}')

        return iou_list_test_case, dice_list_test_case, hit_list_test_case, hau_list_test_case, \
               iou_case, dice_case, hit_case, hau_case
