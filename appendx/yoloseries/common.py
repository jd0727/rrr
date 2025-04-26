import copy
import os
import torch
import torch.nn as nn
from ultralytics.nn import yaml_model_load, parse_model
from ultralytics.utils.torch_utils import intersect_dicts, initialize_weights, scale_img


# class BaseModel(nn.Module):
#     """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""
#
#     def forward(self, x, *args, **kwargs):
#         return self.predict(x, *args, **kwargs)
#
#     def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
#         return self._predict_once(x, profile, visualize, embed)
#
#     def _predict_once(self, x, profile=False, visualize=False, embed=None):
#
#         y, dt, embeddings = [], [], []  # outputs
#         for m in self.model:
#             if m.f != -1:  # if not from previous layer
#                 x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
#             if profile:
#                 self._profile_one_layer(m, x, dt)
#             x = m(x)  # run
#             y.append(x if m.i in self.save else None)  # save output
#
#             if embed and m.i in embed:
#                 embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
#                 if m.i == max(embed):
#                     return torch.unbind(torch.cat(embeddings, 1), dim=0)
#         return x
#
#     def _predict_augment(self, x):
#         return self._predict_once(x)
#
#     def _profile_one_layer(self, m, x, dt):
#         return self
#
#     def fuse(self, verbose=True):
#         return self
#
#     def is_fused(self, thresh=10):
#         return False
#
#     def info(self, detailed=False, verbose=True, imgsz=640):
#         return ''
#
#     def load(self, weights, verbose=True):
#         model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
#         csd = model.float().state_dict()  # checkpoint state_dict as FP32
#         csd = intersect_dicts(csd, self.state_dict())  # intersect
#         self.load_state_dict(csd, strict=False)  # load
#
#     def loss(self, batch, preds=None):
#         """
#         Compute loss.
#
#         Args:
#             batch (dict): Batch to compute loss on
#             preds (torch.Tensor | List[torch.Tensor]): Predictions.
#         """
#         if getattr(self, "criterion", None) is None:
#             self.criterion = self.init_criterion()
#
#         preds = self.forward(batch["img"]) if preds is None else preds
#         return self.criterion(preds, batch)
#
#     def init_criterion(self):
#         """Initialize the loss criterion for the BaseModel."""
#         raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class YoloDetectionModel(nn.Module):
    """YOLOv8 detection model."""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        if self.yaml["backbone"][0][2] == "Silence":
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(copy.deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        ########################################
        self.out_index = self.yaml['out']
        self.save.extend(self.out_index)
        #######################################
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides
        m = self.model[-1]  # Detect()
        self.stride = torch.Tensor([32])

        # Init weights, biases
        initialize_weights(self)

    def forward(self, x, profile=False, visualize=False, embed=None):

        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        ys = [y[i] for i in self.out_index]
        return ys

    def _predict_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return None
