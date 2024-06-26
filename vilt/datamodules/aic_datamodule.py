from vilt.datasets.aic_dataset import AICDataset
from .datamodule_base import BaseDataModule


class AICDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return AICDataset

    @property
    def dataset_cls_no_false(self):
        return AICDataset

    @property
    def dataset_name(self):
        return "aic"
