# # Copyright (c) Facebook, Inc. and its affiliates.
# # Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC
#
# import copy
# import logging
# import random
# import numpy as np
# from typing import List, Union
# import torch
#
# from detectron2.config import configurable
# from detectron2.structures import (
#     BitMasks,
#     Boxes,
#     BoxMode,
#     Instances,
# )
#
# from detectron2.data import detection_utils as utils
# from detectron2.data import transforms as T
#
# from .augmentation import build_augmentation
#
# __all__ = ["YTVISDatasetMapper", "CocoClipDatasetMapper"]
#
#
# def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
#     """
#     Filter out empty instances in an `Instances` object.
#
#     Args:
#         instances (Instances):
#         by_box (bool): whether to filter out instances with empty boxes
#         by_mask (bool): whether to filter out instances with empty masks
#         box_threshold (float): minimum width and height to be considered non-empty
#
#     Returns:
#         Instances: the filtered instances.
#     """
#     assert by_box or by_mask
#     r = []
#     if by_box:
#         r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
#     if instances.has("gt_masks") and by_mask:
#         r.append(instances.gt_masks.nonempty())
#
#     if not r:
#         return instances
#     m = r[0]
#     for x in r[1:]:
#         m = m & x
#
#     instances.gt_ids[~m] = -1
#     return instances
#
#
# def _get_dummy_anno(num_classes):
#     return {
#         "iscrowd": 0,
#         "category_id": num_classes,
#         "id": -1,
#         "bbox": np.array([0, 0, 0, 0]),
#         "bbox_mode": BoxMode.XYXY_ABS,
#         "segmentation": [np.array([0.0] * 6)]
#     }
#
#
# def ytvis_annotations_to_instances(annos, image_size):
#     """
#     Create an :class:`Instances` object used by the models,
#     from instance annotations in the dataset dict.
#
#     Args:
#         annos (list[dict]): a list of instance annotations in one image, each
#             element for one instance.
#         image_size (tuple): height, width
#
#     Returns:
#         Instances:
#             It will contain fields "gt_boxes", "gt_classes", "gt_ids",
#             "gt_masks", if they can be obtained from `annos`.
#             This is the format that builtin models expect.
#     """
#     boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
#     target = Instances(image_size)
#     target.gt_boxes = Boxes(boxes)
#
#     classes = [int(obj["category_id"]) for obj in annos]
#     classes = torch.tensor(classes, dtype=torch.int64)
#     target.gt_classes = classes
#
#     ids = [int(obj["id"]) for obj in annos]
#     ids = torch.tensor(ids, dtype=torch.int64)
#     target.gt_ids = ids
#
#     if len(annos) and "segmentation" in annos[0]:
#         segms = [obj["segmentation"] for obj in annos]
#         masks = []
#         for segm in segms:
#             assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
#                 segm.ndim
#             )
#             # mask array
#             masks.append(segm)
#         # torch.from_numpy does not support array with negative stride.
#         masks = BitMasks(
#             torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
#         )
#         target.gt_masks = masks
#
#     return target
#
#
# class YTVISDatasetMapper:
#     """
#     A callable which takes a dataset dict in YouTube-VIS Dataset format,
#     and map it into a format used by the model.
#     """
#
#     @configurable
#     def __init__(
#         self,
#         is_train: bool,
#         *,
#         augmentations: List[Union[T.Augmentation, T.Transform]],
#         image_format: str,
#         use_instance_mask: bool = False,
#         sampling_frame_num: int = 2,
#         sampling_frame_range: int = 5,
#         sampling_frame_shuffle: bool = False,
#         num_classes: int = 40,
#     ):
#         """
#         NOTE: this interface is experimental.
#         Args:
#             is_train: whether it's used in training or inference
#             augmentations: a list of augmentations or deterministic transforms to apply
#             image_format: an image format supported by :func:`detection_utils.read_image`.
#             use_instance_mask: whether to process instance segmentation annotations, if available
#         """
#         # fmt: off
#         self.is_train               = is_train
#         self.augmentations          = T.AugmentationList(augmentations)
#         self.image_format           = image_format
#         self.use_instance_mask      = use_instance_mask
#         self.sampling_frame_num     = sampling_frame_num
#         self.sampling_frame_range   = sampling_frame_range
#         self.sampling_frame_shuffle = sampling_frame_shuffle
#         self.num_classes            = num_classes
#         # fmt: on
#         logger = logging.getLogger(__name__)
#         mode = "training" if is_train else "inference"
#         logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
#
#     @classmethod
#     def from_config(cls, cfg, is_train: bool = True):
#         augs = build_augmentation(cfg, is_train)
#
#         sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
#         sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
#         sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE
#
#         ret = {
#             "is_train": is_train,
#             "augmentations": augs,
#             "image_format": cfg.INPUT.FORMAT,
#             "use_instance_mask": cfg.MODEL.MASK_ON,
#             "sampling_frame_num": sampling_frame_num,
#             "sampling_frame_range": sampling_frame_range,
#             "sampling_frame_shuffle": sampling_frame_shuffle,
#             "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
#         }
#
#         return ret
#
#     def __call__(self, dataset_dict):
#         """
#         Args:
#             dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.
#
#         Returns:
#             dict: a format that builtin models in detectron2 accept
#         """
#         # TODO consider examining below deepcopy as it costs huge amount of computations.
#         dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#
#         video_length = dataset_dict["length"]
#         if self.is_train:
#             ref_frame = random.randrange(video_length)
#
#             start_idx = max(0, ref_frame-self.sampling_frame_range)
#             end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)
#
#             selected_idx = np.random.choice(
#                 np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
#                 self.sampling_frame_num - 1,
#             )
#             selected_idx = selected_idx.tolist() + [ref_frame]
#             selected_idx = sorted(selected_idx)
#             if self.sampling_frame_shuffle:
#                 random.shuffle(selected_idx)
#         else:
#             selected_idx = range(video_length)
#
#         video_annos = dataset_dict.pop("annotations", None)
#         file_names = dataset_dict.pop("file_names", None)
#
#         if self.is_train:
#             _ids = set()
#             for frame_idx in selected_idx:
#                 _ids.update([anno["id"] for anno in video_annos[frame_idx]])
#             ids = dict()
#             for i, _id in enumerate(_ids):
#                 ids[_id] = i
#
#         dataset_dict["image"] = []
#         dataset_dict["instances"] = []
#         dataset_dict["file_names"] = []
#         for frame_idx in selected_idx:
#             dataset_dict["file_names"].append(file_names[frame_idx])
#
#             # Read image
#             image = utils.read_image(file_names[frame_idx], format=self.image_format)
#             utils.check_image_size(dataset_dict, image)
#
#             aug_input = T.AugInput(image)
#             transforms = self.augmentations(aug_input)
#             image = aug_input.image
#
#             image_shape = image.shape[:2]  # h, w
#             # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
#             # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
#             # Therefore it's important to use torch.Tensor.
#             dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))
#
#             if (video_annos is None) or (not self.is_train):
#                 continue
#
#             # NOTE copy() is to prevent annotations getting changed from applying augmentations
#             _frame_annos = []
#             for anno in video_annos[frame_idx]:
#                 _anno = {}
#                 for k, v in anno.items():
#                     _anno[k] = copy.deepcopy(v)
#                 _frame_annos.append(_anno)
#
#             # USER: Implement additional transformations if you have other types of data
#             annos = [
#                 utils.transform_instance_annotations(obj, transforms, image_shape)
#                 for obj in _frame_annos
#                 if obj.get("iscrowd", 0) == 0
#             ]
#             sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids))]
#
#             for _anno in annos:
#                 idx = ids[_anno["id"]]
#                 sorted_annos[idx] = _anno
#             _gt_ids = [_anno["id"] for _anno in sorted_annos]
#
#             instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
#             instances.gt_ids = torch.tensor(_gt_ids)
#             if instances.has("gt_masks"):
#                 instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
#                 instances = filter_empty_instances(instances)
#             else:
#                 instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
#             dataset_dict["instances"].append(instances)
#
#         return dataset_dict
#
#
# class CocoClipDatasetMapper:
#     """
#     A callable which takes a COCO image which converts into multiple frames,
#     and map it into a format used by the model.
#     """
#
#     @configurable
#     def __init__(
#         self,
#         is_train: bool,
#         *,
#         augmentations: List[Union[T.Augmentation, T.Transform]],
#         image_format: str,
#         use_instance_mask: bool = False,
#         sampling_frame_num: int = 2,
#     ):
#         """
#         NOTE: this interface is experimental.
#         Args:
#             is_train: whether it's used in training or inference
#             augmentations: a list of augmentations or deterministic transforms to apply
#             image_format: an image format supported by :func:`detection_utils.read_image`.
#             use_instance_mask: whether to process instance segmentation annotations, if available
#         """
#         # fmt: off
#         self.is_train               = is_train
#         self.augmentations          = T.AugmentationList(augmentations)
#         self.image_format           = image_format
#         self.use_instance_mask      = use_instance_mask
#         self.sampling_frame_num     = sampling_frame_num
#         # fmt: on
#         logger = logging.getLogger(__name__)
#         mode = "training" if is_train else "inference"
#         logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
#
#     @classmethod
#     def from_config(cls, cfg, is_train: bool = True):
#         augs = build_augmentation(cfg, is_train)
#
#         sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
#
#         ret = {
#             "is_train": is_train,
#             "augmentations": augs,
#             "image_format": cfg.INPUT.FORMAT,
#             "use_instance_mask": cfg.MODEL.MASK_ON,
#             "sampling_frame_num": sampling_frame_num,
#         }
#
#         return ret
#
#     def __call__(self, dataset_dict):
#         """
#         Args:
#             dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
#
#         Returns:
#             dict: a format that builtin models in detectron2 accept
#         """
#         dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#
#         img_annos = dataset_dict.pop("annotations", None)
#         file_name = dataset_dict.pop("file_name", None)
#         original_image = utils.read_image(file_name, format=self.image_format)
#
#         dataset_dict["image"] = []
#         dataset_dict["instances"] = []
#         dataset_dict["file_names"] = [file_name] * self.sampling_frame_num
#         for _ in range(self.sampling_frame_num):
#             utils.check_image_size(dataset_dict, original_image)
#
#             aug_input = T.AugInput(original_image)
#             transforms = self.augmentations(aug_input)
#             image = aug_input.image
#
#             image_shape = image.shape[:2]  # h, w
#             # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
#             # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
#             # Therefore it's important to use torch.Tensor.
#             dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))
#
#             if (img_annos is None) or (not self.is_train):
#                 continue
#
#             _img_annos = []
#             for anno in img_annos:
#                 _anno = {}
#                 for k, v in anno.items():
#                     _anno[k] = copy.deepcopy(v)
#                 _img_annos.append(_anno)
#
#             # USER: Implement additional transformations if you have other types of data
#             annos = [
#                 utils.transform_instance_annotations(obj, transforms, image_shape)
#                 for obj in _img_annos
#                 if obj.get("iscrowd", 0) == 0
#             ]
#             _gt_ids = list(range(len(annos)))
#             for idx in range(len(annos)):
#                 if len(annos[idx]["segmentation"]) == 0:
#                     annos[idx]["segmentation"] = [np.array([0.0] * 6)]
#
#             instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
#             instances.gt_ids = torch.tensor(_gt_ids)
#             if instances.has("gt_masks"):
#                 instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
#                 instances = filter_empty_instances(instances)
#             else:
#                 instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
#             dataset_dict["instances"].append(instances)
#
#         return dataset_dict
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukiunhwang/IFC

import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .augmentation import build_augmentation

__all__ = ["YTVISDatasetMapper", "CocoClipDatasetMapper"]

import mmcv
YTVIS_2019_TRAIN_SRC = mmcv.load('./ytvis_2019_train_src.pkl')
YTVIS_2021_TRAIN_SRC = mmcv.load('./ytvis_2021_train_src.pkl')


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` obiect.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


def _get_dummy_anno(num_classes):
    return {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [np.array([0.0] * 6)]
    }


def ytvis_annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` obiect used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_ids",
            "gt_masks", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obi["bbox"], obi["bbox_mode"], BoxMode.XYXY_ABS) for obi in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obi["category_id"]) for obi in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    ids = [int(obi["id"]) for obi in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.gt_ids = ids

    if len(annos) and "segmentation" in annos[0]:
        segms = [obi["segmentation"] for obi in annos]
        masks = []
        for segm in segms:
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )
        target.gt_masks = masks

    return target


class YTVISDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        rad = random.random()
        if not self.is_train:
            rad=0

        if rad <= 0.5:
            # ============================== Origin ===================================

            dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

            video_length = dataset_dict["length"]
            if self.is_train:
                ref_frame = random.randrange(video_length)

                start_idx = max(0, ref_frame - self.sampling_frame_range)
                end_idx = min(video_length, ref_frame + self.sampling_frame_range + 1)

                selected_idx = np.random.choice(
                    np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame + 1, end_idx))),
                    self.sampling_frame_num - 1,
                )
                selected_idx = selected_idx.tolist() + [ref_frame]
                selected_idx = sorted(selected_idx)
                if self.sampling_frame_shuffle:
                    random.shuffle(selected_idx)
            else:
                selected_idx = range(video_length)

            video_annos = dataset_dict.pop("annotations", None)
            file_names = dataset_dict.pop("file_names", None)

            if self.is_train:
                _ids = set()

                for frame_idx in selected_idx:
                    _ids.update([anno["id"] for anno in video_annos[frame_idx]])
                ids = dict()
                for i, _id in enumerate(_ids):
                    ids[_id] = i

            dataset_dict["image"] = []
            dataset_dict["instances"] = []
            dataset_dict["file_names"] = []
            for frame_idx in selected_idx:
                dataset_dict["file_names"].append(file_names[frame_idx])

                # Read image
                image = utils.read_image(file_names[frame_idx], format=self.image_format)
                utils.check_image_size(dataset_dict, image)

                aug_input = T.AugInput(image)
                transforms = self.augmentations(aug_input)
                image = aug_input.image

                image_shape = image.shape[:2]  # h, w
                # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
                # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
                # Therefore it's important to use torch.Tensor.
                dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

                if (video_annos is None) or (not self.is_train):
                    continue

                # NOTE copy() is to prevent annotations getting changed from applying augmentations
                _frame_annos = []
                for anno in video_annos[frame_idx]:
                    _anno = {}
                    for k, v in anno.items():
                        _anno[k] = copy.deepcopy(v)
                    _frame_annos.append(_anno)

                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(obj, transforms, image_shape)
                    for obj in _frame_annos
                    if obj.get("iscrowd", 0) == 0
                ]
                sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids))]

                for _anno in annos:
                    idx = ids[_anno["id"]]
                    sorted_annos[idx] = _anno
                _gt_ids = [_anno["id"] for _anno in sorted_annos]

                instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
                instances.gt_ids = torch.tensor(_gt_ids)
                if instances.has("gt_masks"):
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                    instances = filter_empty_instances(instances)
                else:
                    instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
                dataset_dict["instances"].append(instances)

            return dataset_dict

        # =============================== copy-paste =================================
        else:
            src = YTVIS_2021_TRAIN_SRC
            # src = YTVIS_2019_TRAIN_SRC
            dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

            src_dataset_dict = random.choice(src)
            if dataset_dict["video_id"] == src_dataset_dict["video_id"]:
                src_dataset_dict = random.choice(src)
                if dataset_dict["video_id"] != src_dataset_dict["video_id"]:
                    src_dataset_dict = src_dataset_dict
            src_dataset_dict = copy.deepcopy(src_dataset_dict)

            video_length = dataset_dict["length"]
            src_video_length = src_dataset_dict["length"]

            if self.is_train:
                ref_frame = random.randrange(video_length)

                start_idx = max(0, ref_frame - self.sampling_frame_range)
                end_idx = min(video_length, ref_frame + self.sampling_frame_range + 1)

                selected_idx = np.random.choice(
                    np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame + 1, end_idx))),
                    self.sampling_frame_num - 1,
                )
                selected_idx = selected_idx.tolist() + [ref_frame]
                selected_idx = sorted(selected_idx)
                if self.sampling_frame_shuffle:
                    random.shuffle(selected_idx)

            # ============================== selected_src_id ===========================================
            if self.is_train:
                ref_src_frame = random.randrange(src_video_length)

                start_idx = max(0, ref_src_frame - self.sampling_frame_range)
                end_idx = min(src_video_length, ref_src_frame + self.sampling_frame_range + 1)

                selected_src_idx = np.random.choice(
                    np.array(list(range(start_idx, ref_src_frame)) + list(range(ref_src_frame + 1, end_idx))),
                    self.sampling_frame_num - 1,
                )
                selected_src_idx = selected_src_idx.tolist() + [ref_src_frame]
                selected_src_idx = sorted(selected_src_idx)
                if self.sampling_frame_shuffle:
                    random.shuffle(selected_src_idx)
            # ============================== selected_src_id ===========================================

            else:
                selected_idx = range(video_length)

            video_annos = dataset_dict.pop("annotations", None)
            file_names = dataset_dict.pop("file_names", None)

            src_video_annos = src_dataset_dict.pop("annotations", None)
            src_file_names = src_dataset_dict.pop("file_names", None)

            if self.is_train:
                _ids = set()
                for frame_idx in selected_idx:
                    _ids.update([anno["id"] for anno in video_annos[frame_idx]])
                ids = dict()
                for i, _id in enumerate(_ids):
                    ids[_id] = i
            # ============================== selected_src_id ===========================================
            if self.is_train:
                _ids_src = set()
                for src_frame_idx in selected_src_idx:
                    _ids_src.update([src_anno["id"] for src_anno in src_video_annos[src_frame_idx]])
                ids_src = dict()
                for i_src, _id_src in enumerate(_ids_src):
                    ids_src[_id_src] = i_src

            dataset_dict["image"] = []
            dataset_dict["instances"] = []
            dataset_dict["file_names"] = []

            for frame_idx in selected_idx:
                dataset_dict["file_names"].append(file_names[frame_idx])

                # Read image
                image = utils.read_image(file_names[frame_idx], format=self.image_format)
                utils.check_image_size(dataset_dict, image)

                aug_input = T.AugInput(image)
                transforms = self.augmentations(aug_input)
                image = aug_input.image

                image_shape = image.shape[:2]  # h, w
                # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
                # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
                # Therefore it's important to use torch.Tensor.
                dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

                if (video_annos is None) or (not self.is_train):
                    continue

                # NOTE copy() is to prevent annotations getting changed from applying augmentations
                _frame_annos = []
                for anno in video_annos[frame_idx]:
                    _anno = {}
                    for k, v in anno.items():
                        _anno[k] = copy.deepcopy(v)
                    _frame_annos.append(_anno)

                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(obi, transforms, image_shape)
                    for obi in _frame_annos
                    if obi.get("iscrowd", 0) == 0
                ]
                sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids))]

                for _anno in annos:
                    idx = ids[_anno["id"]]
                    sorted_annos[idx] = _anno
                _gt_ids = [_anno["id"] for _anno in sorted_annos]

                instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
                instances.gt_ids = torch.tensor(_gt_ids)
                if instances.has("gt_masks"):
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                    instances = filter_empty_instances(instances)
                else:
                    instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
                dataset_dict["instances"].append(instances)

            # ============================== generation src dataset dict ===========================================
            src_dataset_dict["image"] = []
            src_dataset_dict["instances"] = []
            src_dataset_dict["file_names"] = []

            for src_frame_idx in selected_src_idx:
                src_dataset_dict["file_names"].append(src_file_names[src_frame_idx])

                # Read image
                src_image = utils.read_image(src_file_names[src_frame_idx], format=self.image_format)
                utils.check_image_size(src_dataset_dict, src_image)

                aug_input = T.AugInput(src_image)
                transforms = self.augmentations(aug_input)
                src_image = aug_input.image

                src_image_shape = src_image.shape[:2]  # h, w
                # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
                # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
                # Therefore it's important to use torch.Tensor.
                src_dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(src_image.transpose(2, 0, 1))))

                if (src_video_annos is None) or (not self.is_train):
                    continue

                # NOTE copy() is to prevent annotations getting changed from applying augmentations
                _frame_annos = []
                for anno in src_video_annos[src_frame_idx]:
                    _anno = {}
                    for k, v in anno.items():
                        _anno[k] = copy.deepcopy(v)
                    _frame_annos.append(_anno)

                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(obi, transforms, src_image_shape)
                    for obi in _frame_annos
                    if obi.get("iscrowd", 0) == 0
                ]
                src_sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids_src))]

                for _anno in annos:
                    idx = ids_src[_anno["id"]]
                    src_sorted_annos[idx] = _anno
                _gt_ids = [_anno["id"] for _anno in src_sorted_annos]

                instances = utils.annotations_to_instances(src_sorted_annos, src_image_shape, mask_format="bitmask")
                instances.gt_ids = torch.tensor(_gt_ids)
                if instances.has("gt_masks"):
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                    instances = filter_empty_instances(instances)
                else:
                    instances.gt_masks = BitMasks(torch.empty((0, *src_image_shape)))
                src_dataset_dict["instances"].append(instances)

            # # TODO
            dst_len = len(dataset_dict["instances"])
            src_len = len(src_dataset_dict["instances"])

            new_dataset_dict = copy.deepcopy(dataset_dict)
            for i in range(dst_len):
                dst_img = dataset_dict['image'][i]
                dst_img = np.transpose(dst_img.numpy(), (1, 2, 0))
                dst_classes = dataset_dict['instances'][i].gt_classes
                dst_boxes = dataset_dict['instances'][i].gt_boxes
                dst_boxes = np.transpose(dst_boxes.tensor.numpy())
                dst_boxes = np.transpose(dst_boxes, (1, 0))
                dst_masks = dataset_dict['instances'][i].gt_masks
                dst_masks.tensor = dataset_dict['instances'][i].gt_masks.tensor.int()
                dst_masks.tensor = np.transpose(dst_masks.tensor.numpy(), (0, 1, 2))
                dst_ids = dataset_dict['instances'][i].gt_ids.numpy()

                src_img = src_dataset_dict['image'][i]
                src_img = np.transpose(src_img.numpy(), (1, 2, 0))
                src_classes = src_dataset_dict['instances'][i].gt_classes
                src_boxes = src_dataset_dict['instances'][i].gt_boxes
                src_boxes = np.transpose(src_boxes.tensor.numpy())
                src_boxes = np.transpose(src_boxes, (1, 0))
                src_masks = src_dataset_dict['instances'][i].gt_masks
                src_masks.tensor = src_dataset_dict['instances'][i].gt_masks.tensor.int()
                src_masks.tensor = np.transpose(src_masks.tensor.numpy(), (0, 1, 2))
                src_ids = src_dataset_dict['instances'][i].gt_ids.numpy()

                if len(src_boxes) == 0:
                    return dataset_dict

                composed_mask = np.where(np.any(src_masks, axis=0), 1, 0)

                updated_dst_masks = self.get_updated_masks(dst_masks.tensor, composed_mask)
                # mask fusion have problem ...
                new_masks1 = dst_masks
                new_masks1.tensor = updated_dst_masks
                new_masks1.tensor = torch.Tensor(new_masks1.tensor)
                updated_dst_bboxes = new_masks1.get_bounding_boxes()
                assert len(updated_dst_bboxes) == len(updated_dst_masks)

                updated_dst_bboxes = np.transpose(updated_dst_bboxes.tensor.numpy())
                updated_dst_bboxes = np.transpose(updated_dst_bboxes, (1, 0))

                bboxes_inds = np.all(
                    np.abs(
                        (updated_dst_bboxes - dst_boxes)) <= 10,  # bbox_occluded_thr=10
                    axis=-1)
                masks_inds = updated_dst_masks.sum(
                    axis=(1, 2)) > 300  # mask_occluded_thr=300
                valid_inds = bboxes_inds | masks_inds

                img = dst_img * (1 - composed_mask[..., np.newaxis]
                                 ) + src_img * composed_mask[..., np.newaxis]
                bboxes = np.concatenate([updated_dst_bboxes[valid_inds], src_boxes])
                classes = np.concatenate([dst_classes[valid_inds], src_classes])
                masks = np.concatenate([updated_dst_masks[valid_inds], src_masks.tensor])
                ids = np.concatenate([dst_ids[valid_inds], src_ids])

                new_dataset_dict['image'][i] = dataset_dict['image'][i].new_tensor(img).permute(2, 0, 1)
                new_dataset_dict['instances'][i].gt_boxes.tensor = dataset_dict['instances'][i].gt_boxes.tensor.new_tensor(bboxes)
                new_dataset_dict['instances'][i].gt_masks.tensor = dataset_dict['instances'][i].gt_masks.tensor.new_tensor(masks, dtype=bool)
                new_dataset_dict['instances'][i].gt_classes = dataset_dict['instances'][i].gt_classes.new_tensor(classes)
                new_dataset_dict['instances'][i].gt_ids = dataset_dict['instances'][i].gt_ids.new_tensor(ids)

            f1_ids = len(new_dataset_dict['instances'][0].gt_ids)
            f2_ids = len(new_dataset_dict['instances'][1].gt_ids)
            f3_ids = len(new_dataset_dict['instances'][2].gt_ids)
            f4_ids = len(new_dataset_dict['instances'][3].gt_ids)
            f5_ids = len(new_dataset_dict['instances'][4].gt_ids)
            all_ids = max(f1_ids, f2_ids, f3_ids, f4_ids, f5_ids)

            if f1_ids < all_ids:
                empty_ids_1 = all_ids - f1_ids
                image_shape_1 = new_dataset_dict['instances'][0].image_size

                # generation new empty instances
                empty_annos_1 = [_get_dummy_anno(self.num_classes) for _ in range((empty_ids_1))]
                empty_instances_1 = utils.annotations_to_instances(empty_annos_1, image_shape_1, mask_format="bitmask")

                id_empty = []
                for i in range(empty_ids_1):
                    id_empty.append(-1)
                empty_instances_1.gt_ids = torch.tensor(id_empty)

                # for i in range(empty_ids_1):
                cat_boxes = torch.cat([new_dataset_dict['instances'][0].gt_boxes.tensor, empty_instances_1.gt_boxes.tensor], dim=0)
                cat_classes = torch.cat([new_dataset_dict['instances'][0].gt_classes, empty_instances_1.gt_classes], dim=0)
                cat_masks = torch.cat([new_dataset_dict['instances'][0].gt_masks.tensor, empty_instances_1.gt_masks.tensor], dim=0)
                cat_ids = torch.cat([new_dataset_dict['instances'][0].gt_ids, empty_instances_1.gt_ids], dim=0)

                cat_boxes = np.transpose(np.transpose(cat_boxes.numpy()), (1, 0))
                cat_classes = np.transpose(cat_classes.numpy())
                cat_masks = np.transpose(np.transpose(cat_masks.numpy()), (2, 1, 0))
                cat_ids = np.transpose(cat_ids.numpy())

                new_dataset_dict['instances'][0].gt_boxes.tensor = new_dataset_dict['instances'][0].gt_boxes.tensor.new_tensor(cat_boxes)
                new_dataset_dict['instances'][0].gt_classes = new_dataset_dict['instances'][0].gt_classes.new_tensor(cat_classes)
                new_dataset_dict['instances'][0].gt_masks.tensor = new_dataset_dict['instances'][0].gt_masks.tensor.new_tensor(cat_masks, dtype=bool)
                new_dataset_dict['instances'][0].gt_ids = new_dataset_dict['instances'][0].gt_ids.new_tensor(cat_ids)

            if f2_ids < all_ids:
                empty_ids_2 = all_ids - f2_ids
                image_shape_2 = new_dataset_dict['instances'][1].image_size

                # generation new empty instances
                empty_annos_2 = [_get_dummy_anno(self.num_classes) for _ in range((empty_ids_2))]
                empty_instances_2 = utils.annotations_to_instances(empty_annos_2, image_shape_2, mask_format="bitmask")

                id_empty = []
                for i in range(empty_ids_2):
                    id_empty.append(-1)
                empty_instances_2.gt_ids = torch.tensor(id_empty)

                # for i in range(empty_ids_2):
                cat_boxes = torch.cat([new_dataset_dict['instances'][1].gt_boxes.tensor, empty_instances_2.gt_boxes.tensor], dim=0)
                cat_classes = torch.cat([new_dataset_dict['instances'][1].gt_classes, empty_instances_2.gt_classes], dim=0)
                cat_masks = torch.cat([new_dataset_dict['instances'][1].gt_masks.tensor, empty_instances_2.gt_masks.tensor], dim=0)
                cat_ids = torch.cat([new_dataset_dict['instances'][1].gt_ids, empty_instances_2.gt_ids], dim=0)

                cat_boxes = np.transpose(np.transpose(cat_boxes.numpy()), (1, 0))
                cat_classes = np.transpose(cat_classes.numpy())
                cat_masks = np.transpose(np.transpose(cat_masks.numpy()), (2, 1, 0))
                cat_ids = np.transpose(cat_ids.numpy())

                new_dataset_dict['instances'][1].gt_boxes.tensor = new_dataset_dict['instances'][1].gt_boxes.tensor.new_tensor(cat_boxes)
                new_dataset_dict['instances'][1].gt_classes = new_dataset_dict['instances'][1].gt_classes.new_tensor(cat_classes)
                new_dataset_dict['instances'][1].gt_masks.tensor = new_dataset_dict['instances'][1].gt_masks.tensor.new_tensor(cat_masks, dtype=bool)
                new_dataset_dict['instances'][1].gt_ids = new_dataset_dict['instances'][1].gt_ids.new_tensor(cat_ids)

            if f3_ids < all_ids:
                empty_ids_3 = all_ids - f3_ids
                image_shape_3 = new_dataset_dict['instances'][2].image_size

                # generation new empty instances
                empty_annos_3 = [_get_dummy_anno(self.num_classes) for _ in range((empty_ids_3))]
                empty_instances_3 = utils.annotations_to_instances(empty_annos_3, image_shape_3, mask_format="bitmask")

                id_empty = []
                for i in range(empty_ids_3):
                    id_empty.append(-1)
                empty_instances_3.gt_ids = torch.tensor(id_empty)

                # for i in range(empty_ids_1):
                cat_boxes = torch.cat([new_dataset_dict['instances'][2].gt_boxes.tensor, empty_instances_3.gt_boxes.tensor], dim=0)
                cat_classes = torch.cat([new_dataset_dict['instances'][2].gt_classes, empty_instances_3.gt_classes], dim=0)
                cat_masks = torch.cat([new_dataset_dict['instances'][2].gt_masks.tensor, empty_instances_3.gt_masks.tensor], dim=0)
                cat_ids = torch.cat([new_dataset_dict['instances'][2].gt_ids, empty_instances_3.gt_ids], dim=0)

                cat_boxes = np.transpose(np.transpose(cat_boxes.numpy()), (1, 0))
                cat_classes = np.transpose(cat_classes.numpy())
                cat_masks = np.transpose(np.transpose(cat_masks.numpy()), (2, 1, 0))
                cat_ids = np.transpose(cat_ids.numpy())

                new_dataset_dict['instances'][2].gt_boxes.tensor = new_dataset_dict['instances'][2].gt_boxes.tensor.new_tensor(cat_boxes)
                new_dataset_dict['instances'][2].gt_classes = new_dataset_dict['instances'][2].gt_classes.new_tensor(cat_classes)
                new_dataset_dict['instances'][2].gt_masks.tensor = new_dataset_dict['instances'][2].gt_masks.tensor.new_tensor(cat_masks, dtype=bool)
                new_dataset_dict['instances'][2].gt_ids = new_dataset_dict['instances'][2].gt_ids.new_tensor(cat_ids)

            if f4_ids < all_ids:
                empty_ids_4 = all_ids - f4_ids
                image_shape_4 = new_dataset_dict['instances'][3].image_size

                # generation new empty instances
                empty_annos_4 = [_get_dummy_anno(self.num_classes) for _ in range((empty_ids_4))]
                empty_instances_4 = utils.annotations_to_instances(empty_annos_4, image_shape_4, mask_format="bitmask")

                id_empty = []
                for i in range(empty_ids_4):
                    id_empty.append(-1)
                empty_instances_4.gt_ids = torch.tensor(id_empty)

                # for i in range(empty_ids_1):
                cat_boxes = torch.cat([new_dataset_dict['instances'][3].gt_boxes.tensor, empty_instances_4.gt_boxes.tensor], dim=0)
                cat_classes = torch.cat([new_dataset_dict['instances'][3].gt_classes, empty_instances_4.gt_classes], dim=0)
                cat_masks = torch.cat([new_dataset_dict['instances'][3].gt_masks.tensor, empty_instances_4.gt_masks.tensor], dim=0)
                cat_ids = torch.cat([new_dataset_dict['instances'][3].gt_ids, empty_instances_4.gt_ids], dim=0)

                cat_boxes = np.transpose(np.transpose(cat_boxes.numpy()), (1, 0))
                cat_classes = np.transpose(cat_classes.numpy())
                cat_masks = np.transpose(np.transpose(cat_masks.numpy()), (2, 1, 0))
                cat_ids = np.transpose(cat_ids.numpy())

                new_dataset_dict['instances'][3].gt_boxes.tensor = new_dataset_dict['instances'][3].gt_boxes.tensor.new_tensor(cat_boxes)
                new_dataset_dict['instances'][3].gt_classes = new_dataset_dict['instances'][3].gt_classes.new_tensor(cat_classes)
                new_dataset_dict['instances'][3].gt_masks.tensor = new_dataset_dict['instances'][3].gt_masks.tensor.new_tensor(cat_masks, dtype=bool)
                new_dataset_dict['instances'][3].gt_ids = new_dataset_dict['instances'][3].gt_ids.new_tensor(cat_ids)

            if f5_ids < all_ids:
                empty_ids_5 = all_ids - f5_ids
                image_shape_5 = new_dataset_dict['instances'][4].image_size

                # generation new empty instances
                empty_annos_5 = [_get_dummy_anno(self.num_classes) for _ in range((empty_ids_5))]
                empty_instances_5 = utils.annotations_to_instances(empty_annos_5, image_shape_5, mask_format="bitmask")

                id_empty = []
                for i in range(empty_ids_5):
                    id_empty.append(-1)
                empty_instances_5.gt_ids = torch.tensor(id_empty)

                # for i in range(empty_ids_1):
                cat_boxes = torch.cat([new_dataset_dict['instances'][4].gt_boxes.tensor, empty_instances_5.gt_boxes.tensor], dim=0)
                cat_classes = torch.cat([new_dataset_dict['instances'][4].gt_classes, empty_instances_5.gt_classes], dim=0)
                cat_masks = torch.cat([new_dataset_dict['instances'][4].gt_masks.tensor, empty_instances_5.gt_masks.tensor], dim=0)
                cat_ids = torch.cat([new_dataset_dict['instances'][4].gt_ids, empty_instances_5.gt_ids], dim=0)

                cat_boxes = np.transpose(np.transpose(cat_boxes.numpy()), (1, 0))
                cat_classes = np.transpose(cat_classes.numpy())
                cat_masks = np.transpose(np.transpose(cat_masks.numpy()), (2, 1, 0))
                cat_ids = np.transpose(cat_ids.numpy())

                new_dataset_dict['instances'][4].gt_boxes.tensor = new_dataset_dict['instances'][4].gt_boxes.tensor.new_tensor(cat_boxes)
                new_dataset_dict['instances'][4].gt_classes = new_dataset_dict['instances'][4].gt_classes.new_tensor(cat_classes)
                new_dataset_dict['instances'][4].gt_masks.tensor = new_dataset_dict['instances'][4].gt_masks.tensor.new_tensor(cat_masks, dtype=bool)
                new_dataset_dict['instances'][4].gt_ids = new_dataset_dict['instances'][4].gt_ids.new_tensor(cat_ids)


            return new_dataset_dict

    def get_updated_masks(self, masks, composed_mask):
        assert masks.shape[-2:] == composed_mask.shape[-2:], \
            'Cannot compare two arrays of different size'
        masks = np.where(composed_mask, 0, masks)
        return masks

    def get_bboxes(self, result):
        num_masks = len(result)
        boxes = np.zeros((num_masks, 4), dtype=np.float32)
        x_any = result.any(axis=1)
        y_any = result.any(axis=2)
        for idx in range(num_masks):
            x = np.where(x_any[idx, :])[0]
            y = np.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                # use +1 for x_max and y_max so that the right and bottom
                # boundary of instance masks are fully included by the box
                boxes[idx, :] = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1],
                                         dtype=np.float32)
        return boxes


class CocoClipDatasetMapper:
    """
    A callable which takes a COCO image which converts into multiple frames,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        img_annos = dataset_dict.pop("annotations", None)
        file_name = dataset_dict.pop("file_name", None)
        original_image = utils.read_image(file_name, format=self.image_format)

        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = [file_name] * self.sampling_frame_num
        for _ in range(self.sampling_frame_num):
            utils.check_image_size(dataset_dict, original_image)

            aug_input = T.AugInput(original_image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (img_annos is None) or (not self.is_train):
                continue

            _img_annos = []
            for anno in img_annos:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _img_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obi, transforms, image_shape)
                for obi in _img_annos
                if obi.get("iscrowd", 0) == 0
            ]
            _gt_ids = list(range(len(annos)))
            for idx in range(len(annos)):
                if len(annos[idx]["segmentation"]) == 0:
                    annos[idx]["segmentation"] = [np.array([0.0] * 6)]

            instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
            instances.gt_ids = torch.tensor(_gt_ids)
            if instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                instances = filter_empty_instances(instances)
            else:
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            dataset_dict["instances"].append(instances)

        return dataset_dict


