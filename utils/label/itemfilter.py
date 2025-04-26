from .imgitem import *


class LabelFilter(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, label) -> bool:
        pass


class ItemFilter(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, item) -> bool:
        pass


class ItemFilterValueEqual(ItemFilter):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, item) -> bool:
        for key, value in self.kwargs.items():
            if not isinstance(item, dict) or key not in item.keys() or not value == item[key]:
                return False
        return True


class ItemFilterBasic(ItemFilterValueEqual):

    def __init__(self, cls_names=None, thres=-1, **kwargs):
        self.cls_names = cls_names
        self.thres = thres
        ItemFilterValueEqual.__init__(self, **kwargs)

    def __call__(self, item):
        if not ItemFilterValueEqual.__call__(self, item):
            return False
        if self.cls_names is not None and item['name'] not in self.cls_names:
            return False
        if self.thres > 0 and item.measure < self.thres:
            return False
        return True


class ItemFilterCind(ItemFilter):
    def __init__(self, cinds):
        self.cinds = cinds

    def __call__(self, item) -> bool:
        return item.category._cindN in self.cinds


class ItemFilterAttributeEqual(ItemFilter):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, item) -> bool:
        for key, value in self.kwargs.items():
            if not hasattr(item, key) or not value == getattr(item, key):
                return False
        return True
