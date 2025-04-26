from .define import OptimizerBuildActor
from utils import *


# <editor-fold desc='优化器构建'>

class SGDBuilder(OptimizerBuildActor):

    def __init__(self, lr: float = 0.001, momentum: float = 0.937, dampening: float = 0, weight_decay: float = 5e-4,
                 module_name: Optional[str] = None):
        OptimizerBuildActor.__init__(self, module_name=module_name)
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.lr = lr

    def act_build_optimizer(self, module: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            params=filter(lambda x: x.requires_grad, module.parameters()),
            momentum=self.momentum, weight_decay=self.weight_decay, lr=self.lr, dampening=self.dampening, )


class GroupSGDBuilder(OptimizerBuildActor):
    def __init__(self, lr: float = 0.001, momentum: float = 0.937, dampening: float = 0, weight_decay: float = 5e-4,
                 module_name: Optional[str] = None):
        OptimizerBuildActor.__init__(self, module_name=module_name)
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.lr = lr

    def act_build_optimizer(self, module: nn.Module) -> torch.optim.Optimizer:
        group_bias, group_weight, group_weight_ndcy = [], [], []
        for name, para in module.named_parameters():
            # print(name)
            if 'bias' in name:  # bias (no decay)
                group_bias.append(para)
            elif 'weight' in name and 'bn' in name:  # weight (no decay)
                group_weight_ndcy.append(para)
            else:
                group_weight.append(para)

        optimizer = torch.optim.SGD(group_bias, lr=self.lr, momentum=self.momentum, nesterov=True)
        optimizer.add_param_group({'params': group_weight, 'weight_decay': self.weight_decay})
        optimizer.add_param_group({'params': group_weight_ndcy, 'weight_decay': 0.0})

        return optimizer


class RMSpropBuilder(OptimizerBuildActor):

    def __init__(self, lr: float = 0.01, alpha: float = 0.99, weight_decay: float = 0, momentum: float = 0,
                 module_name: Optional[str] = None):
        OptimizerBuildActor.__init__(self, module_name=module_name)

        self.momentum = momentum
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.lr = lr

    def act_build_optimizer(self, module: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.RMSprop(
            params=filter(lambda x: x.requires_grad, module.parameters()),
            momentum=self.momentum, weight_decay=self.weight_decay, lr=self.lr, alpha=self.alpha)


class AdamBuilder(OptimizerBuildActor):

    def __init__(self, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-5, module_name: Optional[str] = None):
        OptimizerBuildActor.__init__(self, module_name=module_name)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.lr = lr

    def act_build_optimizer(self, module: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params=filter(lambda x: x.requires_grad, module.parameters()),
            eps=self.eps, weight_decay=self.weight_decay, lr=self.lr, betas=self.betas)


class AdamWBuilder(OptimizerBuildActor):

    def __init__(self, lr: float = 0.001, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-5,
                 module_name: Optional[str] = None):
        OptimizerBuildActor.__init__(self, module_name=module_name)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.lr = lr

    def act_build_optimizer(self, module: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params=filter(lambda x: x.requires_grad, module.parameters()),
            eps=self.eps, weight_decay=self.weight_decay, lr=self.lr, betas=self.betas)


# </editor-fold>


class RepSGD(torch.optim.SGD):

    def __init__(self, grad_sclr_mapper: dict, params,
                 lr: float, momentum: float = 0, dampening: float = 0, weight_decay: float = 0, nesterov: bool = False):
        torch.optim.SGD.__init__(self, params, lr, momentum, dampening=dampening, weight_decay=weight_decay,
                                 nesterov=nesterov)
        self.grad_sclr_mapper = grad_sclr_mapper

    @torch.no_grad()
    def step(self, closure=None):
        for sub_module, grad_sclr in self.grad_sclr_mapper.items():
            weight = sub_module.weight
            weight.grad.data = weight.grad.data * grad_sclr.to(weight.grad.device)  # Note: multiply here

        loss = super(RepSGD, self).step(closure=closure)

        return loss


class RepSGDBuilder(OptimizerBuildActor):

    def __init__(self, lr: float = 0.001, momentum: float = 0.8, dampening: float = 0, weight_decay: float = 1e-5,
                 nesterov: bool = True, module_name: Optional[str] = None):
        OptimizerBuildActor.__init__(self, module_name=module_name)
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.lr = lr
        self.nesterov = nesterov

    def act_build_optimizer(self,  module: nn.Module) -> torch.optim.Optimizer:
        grad_sclr_mapper = _get_conv_grad_mapper(module)
        return RepSGD(grad_sclr_mapper=grad_sclr_mapper,
                      params=filter(lambda x: x.requires_grad, module.parameters()),
                      momentum=self.momentum, weight_decay=self.weight_decay, lr=self.lr, dampening=self.dampening,
                      nesterov=self.nesterov)


def _get_conv_grad_mapper(module: nn.Module, scale_base: float = 0.9, scale_shct: float = 2.5):
    grad_sclr_mapper = {}
    for name, sub_module in module.named_children():
        if isinstance(sub_module, nn.Conv2d):
            kernel_size = sub_module.kernel_size
            groups = sub_module.groups
            out_channels = sub_module.out_channels
            in_channels = sub_module.in_channels
            if kernel_size[0] * kernel_size[1] > 1:
                grad_sclr = torch.full_like(sub_module.weight, fill_value=scale_base)
                inds_out = np.arange(out_channels)
                inds_in = inds_out % (in_channels // groups)
                grad_sclr[inds_out, inds_in, kernel_size[0] // 2, kernel_size[1] // 2] = scale_shct + scale_base
                grad_sclr_mapper[sub_module] = grad_sclr
        sub_grad_sclr_mapper = _get_conv_grad_mapper(sub_module)
        grad_sclr_mapper.update(sub_grad_sclr_mapper)
    return grad_sclr_mapper


# <editor-fold desc='训练原型'>
class OptimazerManager():

    def __init__(self, ):
        self.optimizers = OrderedDict()


    @property
    def learning_rates(self) -> OrderedDict[str, List[float]]:
        lr_dct = OrderedDict()
        for name_opt, optimizer in self.optimizers.items():
            lr_dct[name_opt] = [pg['lr'] for pg in optimizer.param_groups]
        return lr_dct

    @property
    def optimizer_attr(self) -> OrderedDict[str, List]:
        attrs = OrderedDict()
        for name_opt, optimizer in self.optimizers.items():
            attrs[name_opt] = optimizer.param_groups
        return attrs

    def optimizer_attr_set(self, value, module_name: Optional[str] = None,
                           group_index: Optional[int] = None, attr_name='lr'):
        for name_opt, optimizer in self.optimizers.items():
            if module_name is not None and not module_name == name_opt:
                continue
            for k, param_group in enumerate(optimizer.param_groups):
                if group_index is not None and not group_index == k:
                    continue
                param_group[attr_name] = value
        return self

    def optimizer_lr_set(self, learning_rate, module_name: Optional[str] = None,
                         group_index: Optional[int] = None, ):
        return self.optimizer_attr_set(learning_rate, module_name, group_index)

# </editor-fold>
