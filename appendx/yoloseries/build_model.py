import torch

from appendx.yoloseries.common import YoloDetectionModel

if __name__ == '__main__':
    # wei_pth = 'D:\DeskTop\mim\yoloseries-pretrain/yolo11l.pt'
    # model = YOLO(wei_pth)

    cfg_pth = 'D:\Programs\Python\Rebuild//appendx//yoloseries//yoloseries.yaml'
    # cfg_pth = 'D:\DeskTop\mim\yoloseries-main\cfg//models//11//yolo11l.yaml'
    model = YoloDetectionModel(cfg_pth)

    # path = model.export(format="onnx")
    # print(path)

    x = torch.rand(1, 3, 640, 640)
    results = model(x)
    print(results)
