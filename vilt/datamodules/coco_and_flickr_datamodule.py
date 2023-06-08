from vilt.datasets import CocoAndFlickrDataset
from .datamodule_base import BaseDataModule


class CocoAndFlickrDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CocoAndFlickrDataset

    @property
    def dataset_cls_no_false(self):
        return CocoAndFlickrDataset

    @property
    def dataset_name(self):
        return "coco_and_flickr"
