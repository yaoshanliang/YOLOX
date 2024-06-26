# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 3
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = '/gpfs/work/cpt/shanliangyao19/dataset/USVTrack/dancetrack'
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.test_ann = "test.json"
        
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.random_size = (18, 32)
        self.max_epoch = 300
        self.print_interval = 100
        self.eval_interval = 1
        self.test_conf = 0.1
        self.nmsthre = 0.7
        self.no_aug_epochs = 1
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

        # tracking params for Hybrid-SORT
        self.ckpt = "/gpfs/work/cpt/shanliangyao19/code/HybridSORT/YOLOX_outputs/yolox_m_dancetrack_test_hybrid_sort/best_ckpt.pth.tar"
        # self.ckpt = "/gpfs/work/cpt/shanliangyao19/code/yolox/YOLOX_outputs/USVTrack_0508/best_ckpt.pth"
        # self.use_byte = True
        # self.dataset = "dancetrack"
        # self.inertia = 0.05
        # self.iou_thresh = 0.15
        # self.asso = "Height_Modulated_IoU"
        # self.TCM_first_step = True
        # self.TCM_byte_step = True
        # self.TCM_first_step_weight = 1.5
        # self.TCM_byte_step_weight = 1.0
        # self.hybrid_sort_with_reid = False

    # def get_dataset(self, cache: bool, cache_type: str = "ram"):
    #     from yolox.data import COCODataset, TrainTransform

    #     return COCODataset(
    #         data_dir=self.data_dir,
    #         # image_sets=[('train')],
    #         json_file=self.train_ann,
    #         img_size=self.input_size,
    #         preproc=TrainTransform(
    #             max_labels=50,
    #             flip_prob=self.flip_prob,
    #             hsv_prob=self.hsv_prob),
    #         cache=cache,
    #         cache_type=cache_type,
    #     )

    # def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
    #     """
    #     Get dataset according to cache and cache_type parameters.
    #     Args:
    #         cache (bool): Whether to cache imgs to ram or disk.
    #         cache_type (str, optional): Defaults to "ram".
    #             "ram" : Caching imgs to ram for fast training.
    #             "disk": Caching imgs to disk for fast training.
    #     """
    #     from yolox.data import COCODataset, TrainTransform

    #     return COCODataset(
    #         data_dir=self.data_dir,
    #         json_file=self.train_ann,
    #         img_size=self.input_size,
    #         preproc=TrainTransform(
    #             max_labels=50,
    #             flip_prob=self.flip_prob,
    #             hsv_prob=self.hsv_prob
    #         ),
    #         cache=cache,
    #         cache_type=cache_type,
    #     )

    # def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
    #     from yolox.data import (
    #         COCODataset,
    #         TrainTransform,
    #         YoloBatchSampler,
    #         DataLoader,
    #         InfiniteSampler,
    #         MosaicDetection,
    #     )

    #     dataset = COCODataset(
    #         data_dir=self.data_dir,
    #         json_file=self.train_ann,
    #         name='train',
    #         img_size=self.input_size,
    #         # preproc=TrainTransform(
    #         #     rgb_means=(0.485, 0.456, 0.406),
    #         #     std=(0.229, 0.224, 0.225),
    #         #     max_labels=500,
    #         # ),
    #     )

    #     dataset = MosaicDetection(
    #         dataset,
    #         mosaic=not no_aug,
    #         img_size=self.input_size,
    #         # preproc=TrainTransform(
    #         #     rgb_means=(0.485, 0.456, 0.406),
    #         #     std=(0.229, 0.224, 0.225),
    #         #     max_labels=1000,
    #         # ),
    #         degrees=self.degrees,
    #         translate=self.translate,
    #         # scale=self.scale,
    #         shear=self.shear,
    #         # perspective=self.perspective,
    #         enable_mixup=self.enable_mixup,
    #     )

    #     self.dataset = dataset

    #     if is_distributed:
    #         batch_size = batch_size // dist.get_world_size()

    #     sampler = InfiniteSampler(
    #         len(self.dataset), seed=self.seed if self.seed else 0
    #     )

    #     batch_sampler = YoloBatchSampler(
    #         sampler=sampler,
    #         batch_size=batch_size,
    #         drop_last=False,
    #         input_dimension=self.input_size,
    #         mosaic=not no_aug,
    #     )

    #     dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
    #     dataloader_kwargs["batch_sampler"] = batch_sampler
    #     train_loader = DataLoader(self.dataset, **dataloader_kwargs)

    #     return train_loader

    # def get_eval_loader(self, batch_size, is_distributed, testdev=False, run_tracking=False):   # [hgx0411] dataloader related
    #     from yolox.data import COCODataset, ValTransform
        
    #     if testdev:
    #         valdataset = COCODataset(
    #             data_dir=os.path.join(self.data_dir, "annotations"),
    #             json_file=self.test_ann,
    #             img_size=self.test_size,
    #             name='test',
    #             preproc=ValTransform(
    #                 rgb_means=(0.485, 0.456, 0.406),
    #                 std=(0.229, 0.224, 0.225),
    #             ),
    #             run_tracking=run_tracking
    #         )
    #     else:
    #         valdataset = COCODataset(
    #             data_dir=os.path.join(self.data_dir, "annotations"),
    #             json_file=self.val_ann,
    #             img_size=self.test_size,
    #             name='val',
    #             preproc=ValTransform(
    #                 rgb_means=(0.485, 0.456, 0.406),
    #                 std=(0.229, 0.224, 0.225),
    #             ),
    #             run_tracking=run_tracking
    #         )

    #     if is_distributed:
    #         batch_size = batch_size // dist.get_world_size()
    #         sampler = torch.utils.data.distributed.DistributedSampler(
    #             valdataset, shuffle=False
    #         )
    #     else:
    #         sampler = torch.utils.data.SequentialSampler(valdataset)

    #     dataloader_kwargs = {
    #         "num_workers": self.data_num_workers,
    #         "pin_memory": True,
    #         "sampler": sampler,
    #     }
    #     dataloader_kwargs["batch_size"] = batch_size
    #     val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

    #     return val_loader

