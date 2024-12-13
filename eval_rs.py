import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig
import transformers
from model.evf_sam import EvfSamModel
import os
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from utils.dataset_rs import collate_fn, ValDataset
from functools import partial
import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import json
class Config(transformers.PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # 调用父类构造函数
        for key, value in kwargs.items():
            setattr(self, key, value)

def load_config_from_json(file_path):
    with open(file_path, 'r') as f:
        return Config(**json.load(f))

def parse_args(args):
    parser = argparse.ArgumentParser(description="EVF eval")
    parser.add_argument("--version")
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument("--val_dataset", default="refcoco|unc|testA", type=str, 
                        choices=["refcoco|unc|val", "refcoco|unc|testA", "refcoco|unc|testB", 
                                 "refcoco+|unc|val", "refcoco+|unc|testA", "refcoco+|unc|testB", 
                                 "refcocog|umd|val", "refcocog|umd|test"])
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--model_type", default="ori", choices=["ori", "effi", "sam2"])
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    dist.init_process_group('nccl', init_method="env://")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        padding_side="right",
        use_fast=False,
    )

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )
    config = load_config_from_json("/mnt/vos-xt8an8kg/llm/code/rs/EVF-SAM/config/config.json")
    if args.model_type=="ori":
        from model.evf_sam import EvfSamModel
        # model = EvfSamModel.from_pretrained(args.version, low_cpu_mem_usage=True, **kwargs)
        model = EvfSamModel(config, **kwargs)
        state_dict = torch.load("/mnt/vos-xt8an8kg/llm/code/rs/EVF-SAM/runs/evf-sam2-RS-epoch40/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    elif args.model_type=="sam2":
        from model.evf_sam2_rs import EvfSam2Model # 这里改成了rs版本
        model = EvfSam2Model(config, **kwargs)
        state_dict = torch.load("/mnt/vos-xt8an8kg/llm/code/rs/EVF-SAM/runs/evf-sam2-RS-epoch40/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    elif args.model_type=="effi":
        from model.evf_effisam import EvfEffiSamModel
        model = EvfEffiSamModel.from_pretrained(args.version, low_cpu_mem_usage=True, **kwargs)

    if (not args.load_in_4bit) and (not args.load_in_8bit):
        model = model.cuda()

    model = DistributedDataParallel(model, device_ids=[rank])
    model.eval()

    val_dataset = ValDataset(
        args.dataset_dir,
        args.val_dataset,
        args.image_size,
        model_type=args.model_type
    )
    sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False, rank=rank)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        sampler=sampler,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            local_rank=rank,
        ),
    )
    args.log_dir = "runs"
    os.makedirs(args.log_dir, exist_ok=True)
    giou, ciou = validate(val_loader, model, args)
    if rank==0:
        print(args.val_dataset)
        print("giou{:.3f}_ciou{:.3f}".format(giou, ciou))
    dist.destroy_process_group()

def validate(val_loader, model, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    num = 0
    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_evf"] = input_dict["images_evf"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_evf"] = input_dict["images_evf"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_evf"] = input_dict["images_evf"].float()

        with torch.no_grad():
            output_dict = model(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        print("len of pred_masks",len(pred_masks))
        
        assert len(pred_masks) == 1
        intersection, union, acc_iou = 0.0, 0.0, 0.0
        os.makedirs('results_vis/binary_masks', exist_ok=True)
        os.makedirs('results_vis/binary_outputs', exist_ok=True)
        for idx, (mask_i, output_i) in enumerate(zip(masks_list, output_list)):
            from PIL import Image
            print(mask_i.shape)
            print(output_i.shape)
            binary_mask = (mask_i.cpu().numpy() > 0.5).astype(np.uint8) * 255
            binary_output = (output_i.cpu().numpy() > 0.5).astype(np.uint8) * 255

            # 转换为PIL图像
            binary_mask_image = Image.fromarray(binary_mask)  
            binary_output_image = Image.fromarray(binary_output)  

            # 保存
            binary_mask_image.save(f'results_vis/binary_masks/mask_{num}.png')
            binary_output_image.save(f'results_vis/binary_outputs/output_{num}.png')
            
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            
            
            intersection += intersection_i
            # print(intersection_i)
            # print(intersection)
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-10)
            print(acc_iou)
            # 把 acc_iou 保存到val1.txt文件中
            # with open("val_data_fintune_beit.txt", "a") as f:
            #     f.write(str(acc_iou[1].cpu().numpy().item()) + "\n")
            #     f.write("\n\n")
            # f.close()
            acc_iou[union_i == 0] += 1.0  # no-object target
            
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        # print(acc_iou)
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=masks_list.shape[0])
        num += 1
    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    return giou, ciou

if __name__ == "__main__":
    main(sys.argv[1:])