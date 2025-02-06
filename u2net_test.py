import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model.u2net import U2NET, U2NETP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_path, out_data, d_dir):
    # NCHW  1 C H W ---> C H W
    out_data = out_data.squeeze()
    out_data = out_data.cpu().data.numpy()
    out_data = out_data * 255
    ori_img = cv2.imread(image_path)
    mask_img = cv2.resize(
        out_data,
        dsize=(ori_img.shape[1], ori_img.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    # 一个通道转成3个通道
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(d_dir + os.path.basename(image_path), mask_img)


def model_test():
    # --------- 1. get image path and name ---------
    model_name = "u2netp"  # u2netp
    image_dir = "test_data/"
    prediction_dir = "test_result/"
    model_dir = "saved_models/u2netp-184-2000-final.pt"

    img_name_list = glob.glob(image_dir + "*")
    print(img_name_list)

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]),
    )
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1
    )

    # --------- 3. model define ---------
    if model_name == "u2net":
        net = U2NET(3, 1).to(device)
    elif model_name == "u2netp":
        net = U2NETP(3, 1).to(device)

    # 加载训练好的模型参数
    net.load_state_dict(torch.load(model_dir, map_location=device))
    # 开启测试模式
    net.eval()

    # --------- 4. inference for each image ---------
    for i, (img, _) in enumerate(test_salobj_dataloader):
        img = img.type(torch.FloatTensor)
        img = img.to(device)

        d1, d2, d3, d4, d5, d6, d7 = net(img)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(img_name_list[i], pred, prediction_dir)


if __name__ == "__main__":
    model_test()
