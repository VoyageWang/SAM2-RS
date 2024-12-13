import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask

from model.segment_anything.utils.transforms import ResizeLongestSide

from .grefer import G_REFER
from .refer_rs import REFER
from torchvision import transforms
from utils.dataset_rs import Resize


class RRSISSegDataset(torch.utils.data.Dataset):
    # 这个是对于sam的图像预处理方法的一些值
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024 # 这个是对于sam来说的
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224, # 这个是对于clip的或者这个地方是对于beit的
        num_classes_per_sample: int = 3,
        rsris_seg_data="rrsisd",
        model_type="ori",
        transform=Resize(1024),
    ):
        print(type(transform))
        print(transform)
        if model_type=="ori":
            assert isinstance(transform, ResizeLongestSide)
        else: # sam2的话直接就是resize  符合issue中的 https://github.com/hustvl/EVF-SAM/issues/26
            assert isinstance(transform, Resize)
        self.model_type = model_type
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.precision = precision
        self.transform = transform
        self.image_preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), interpolation=3, antialias=None), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        DATA_DIR = base_image_dir # 直接就指向了base_image_dir
        self.refer_seg_ds_list = [rsris_seg_data]  # RSRIS-D 后面再去实现rrsis 最原始的那个
        self.refer_seg_data = {}
        self.total_images = 0
        
        
        splitBy = "unc" # 遥感这个数据集直接限制在unc就行
        ds = "rrsisd"
        refer_api = REFER(DATA_DIR, ds, splitBy)
        
        ref_ids_train = refer_api.getRefIds(split="train") # 在这里实现的时候我就只先把train写在这里，然后再在外面的时候写成val
        images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
        self.total_images += len(images_ids_train)
        refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)

        refer_seg_ds = {}
        refer_seg_ds["images"] = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)

        for item in loaded_images:
            item = item.copy()
           
            item["file_name"] = os.path.join(
                DATA_DIR, "images/rrsisd/JPEGImages", item["file_name"]
            )
            refer_seg_ds["images"].append(item)
        refer_seg_ds["annotations"] = refer_api.Anns  # anns_train

        print(
            "dataset {} (refs {}) (train split) has {} images and {} annotations.".format(
                ds,
                splitBy,
                len(refer_seg_ds["images"]),
                len(refer_seg_ds["annotations"]),
            )
        )

        img2refs = {}
        for ref in refs_train:
            image_id = ref["image_id"]
            img2refs[image_id] = img2refs.get(image_id, []) + [
                ref,
            ]
        refer_seg_ds["img2refs"] = img2refs
        self.refer_seg_data[ds] = refer_seg_ds

    def __len__(self):
        return self.total_images

    def preprocess(self, x: torch.Tensor) -> torch.Tensor: # sam
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        if self.model_type=="ori": # sam就会这样做，如果说sam2的话，就不会做
            # Pad
            h, w = x.shape[-2:]
            padh = self.img_size - h
            padw = self.img_size - w
            x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ds = random.randint(0, len(self.refer_seg_ds_list) - 1)
        ds = self.refer_seg_ds_list[ds]
        refer_seg_ds = self.refer_seg_data[ds]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.__getitem__(0)

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_classes = sampled_sents
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # crop augmentation
        # h,w,_ = image.shape
        # left, right = int(random.uniform(0,0.05) * w), int(random.uniform(0.95,1) * w)
        # top, bottom = int(random.uniform(0,0.05) * h), int(random.uniform(0.95,1) * h)
        # image = image[top:bottom, left:right]

        # preprocess image for evf
        image_evf = self.image_preprocessor(image)

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        flag = False
        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = mask.frPyObjects(
                                    ann["segmentation"],
                                    image_info["height"],
                                    image_info["width"],
                                )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(
                m, axis=2
            )  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8

            masks.append(m)

        masks = np.stack(masks, axis=0)
        # masks = masks[:, top:bottom, left:right]

        # if ds == 'grefcoco' and flag:
        #     import shutil
        #     image_name = image_path.split("/")[-1]
        #     save_dir = os.path.join("/group/30042/xlai/LISA_refactor_final/debug", image_name.split(".")[0])
        #     os.makedirs(save_dir, exist_ok=True)
        #     shutil.copy(image_path, save_dir)
        #     for i in range(masks.shape[0]):
        #         cv2.imwrite(os.path.join(save_dir, "{}_{}_{}.jpg".format(image_name, i, sampled_classes[i])), masks[i].astype(np.int32) * 100)

        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            image_path,
            image,
            image_evf,
            masks,
            label,
            resize,
            sampled_classes,
        )
