import numpy as np
from PIL import Image


def compute_iou(pred, label, num_classes):
    iou = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        label_inds = label == cls
        intersection = np.logical_and(pred_inds, label_inds).sum()
        union = np.logical_or(pred_inds, label_inds).sum()
        if union == 0:
            iou.append(np.nan)  # 忽略没有出现过的类
        else:
            iou.append(intersection / union)
    return np.nanmean(iou)


def compute_miou(preds, labels, num_classes):
    miou_list = []
    for pred, label in zip(preds, labels):
        miou_list.append(compute_iou(pred, label, num_classes))
    return np.mean(miou_list)


def convert_numpy(preds_path):
    Img = Image.open(preds_path)
    pred = np.array(Img)
    return pred


def label_convert_numpy(labels_path):
    Img = Image.open(labels_path)
    label = np.array(Img)
    return label


if __name__ == "__main__":
    # 示例用法
    preds_path = "E:\\vscode\\code\check\\test_result\\171206_055300134_Camera_6.jpg"  # 预测结果列表
    labels_path = "E:\\vscode\\code\\check\\test_label_mask\\171206_055300134_Camera_6_bin.png"  # 实际标签列表
    num_classes = 1  # 假设有3个类

    preds = convert_numpy(preds_path=preds_path)
    labels = label_convert_numpy(labels_path=labels_path)

    miou = compute_miou(preds, labels, num_classes)
    print(f"Mean IoU: {miou}")
