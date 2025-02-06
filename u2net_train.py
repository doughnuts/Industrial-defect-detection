import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import glob
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from torch.utils.tensorboard import SummaryWriter

from model.u2net import U2NET, U2NETP

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(reduction="mean")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss


def main():
    # ------- 2. set the directory of training dataset --------
    model_name = "u2netp"  #'u2netp'
    # 数据集的路径
    tra_image_dir = "./data/images"
    tra_label_dir = "./data/labels"

    model_path = "saved_models/" + model_name + "-184" + "-2000" + "-final" + ".pt"
    summerWriter = SummaryWriter("logs")
    tra_img_name_list = glob.glob(tra_image_dir + "/*")
    tra_label_name_list = glob.glob(tra_label_dir + "/*")
    print(tra_img_name_list)
    print(tra_label_name_list)
    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_label_name_list))
    print("---")

    # 数据增样
    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_label_name_list,
        transform=transforms.Compose(
            [RescaleT(320), RandomCrop(288), ToTensorLab(flag=0)]
        ),
    )
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=8, shuffle=False)  # 改了

    # ------- 3. define model --------
    # define the net
    if model_name == "u2net":
        net = U2NET(3, 1).to(device)
    elif model_name == "u2netp":
        net = U2NETP(3, 1).to(device)

    # 加载预训练权重
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print("load model success!!!!!!")

    # ------- 4. define optimizer --------
    optimizer = optim.Adam(net.parameters())
    # ------- 5. training process --------
    print("---start training...")

    for epoch in range(0, 50):
        net.train()
        train_loss = 0.0
        for i, (img, label) in enumerate(salobj_dataloader):
            img = img.type(torch.FloatTensor)
            label = label.type(torch.FloatTensor)
            img, label = img.to(device), label.to(device)

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(img)
            loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label)
            # y zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if i % 10 == 0:
                print("train_loss:{0},epoch:{1}".format(loss.item(), epoch))
        train_avg_loss = train_loss / len(salobj_dataloader)
        summerWriter.add_scalar("train_loss", train_avg_loss, epoch)
        # 将训练的输出进行保存
        torch.save(net.state_dict(), model_path)
        print("参数已保存！")


if __name__ == "__main__":
    main()
