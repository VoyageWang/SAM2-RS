import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from torch.utils.tensorboard import SummaryWriter

from model.evf_sam import EvfSamModel
from model.evf_sam2 import EvfSam2Model
from utils.dataset_rs import HybridDataset, ValDataset, collate_fn
from utils.utils import (AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from utils.dataset_rs import Resize
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
    parser = argparse.ArgumentParser(description="EVF-SAM Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="/mnt/vos-xt8an8kg/llm/code/EVF-SAM/evf-sam2_weight/", help="<path to evf-sam>"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--model_type", default="sam2", choices=["ori", "effi", "sam2"])
    # parser.add_argument("--lora_r", default=8, type=int)
    # parser.add_argument(
    #     "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    # )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="refcoco|unc|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="evf-sam", type=str)
    parser.add_argument("--epochs", default=60, type=int)
    parser.add_argument("--steps_per_epoch", default=250, type=int)
    parser.add_argument(
        "--batch_size", default=32, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=2,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    # parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=1.0, type=float)
    parser.add_argument("--bce_loss_weight", default=1.0, type=float)
    # parser.add_argument("--lora_alpha", default=16, type=int)
    # parser.add_argument("--lora_dropout", default=0.05, type=float)
    # parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="/mnt/vos-xt8an8kg/llm/code/rs/EVF-SAM/model_weights/sam2/sam2_hiera_large.pt", type=str)
    parser.add_argument("--encoder_pretrained", default="/mnt/vos-xt8an8kg/llm/code/rs/EVF-SAM/model_weights/beit/beit3_large_patch16_224.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--train_prompt_encoder", action="store_true", default=True)
    parser.add_argument("--config_json_path", default="/mnt/vos-xt8an8kg/llm/code/rs/EVF-SAM/config/config.json", type=str)
    
    # parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    # parser.add_argument(
    #     "--conv_type",
    #     default="llava_v1",
    #     type=str,
    #     choices=["llava_v1", "llava_llama_2"],
    # )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        padding_side="right",
        use_fast=False,
    )

    # mdoel evf-sam
    
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype, "dice_loss_weight": args.dice_loss_weight, 
            "bce_loss_weight": args.bce_loss_weight, "train_mask_decoder": args.train_mask_decoder, 
            "train_prompt_encoder": args.train_prompt_encoder, "vision_pretrained": args.vision_pretrained,
            "encoder_pretrained": args.encoder_pretrained}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": transformers.BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": transformers.BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )
        
    config = load_config_from_json(args.config_json_path)
    
    if args.model_type=="ori":
        from model.evf_sam import EvfSamModel
        model = EvfSamModel(config, **kwargs)
        
    # 下面的先不管
    elif args.model_type=="effi":
        from model.evf_effisam import EvfEffiSamModel
        model = EvfEffiSamModel.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )
    elif args.model_type=="sam2":
        from model.evf_sam2_rs import EvfSam2Model # 这里改成了rs版本
        model = EvfSam2Model(config, **kwargs)
        
    
    # if not args.eval_only:
    #     model.get_model().initialize_lisa_modules(model.get_model().config)  # 这个地方不知道是什么意思，留在这里

    # 这里不需要去 要求是怎么 去冻住，在model中本身去实现了已经
    # for p in vision_tower.parameters():
    #     p.requires_grad = False
    # for p in model.get_model().mm_projector.parameters():
    #     p.requires_grad = False

    # conversation_lib.default_conversation = conversation_lib.conv_templates[
    #     args.conv_type
    # ]

    # lora_r = args.lora_r
    # if lora_r > 0:

    #     def find_linear_layers(model, lora_target_modules):
    #         cls = torch.nn.Linear
    #         lora_module_names = set()
    #         for name, module in model.named_modules():
    #             if (
    #                 isinstance(module, cls)
    #                 and all(
    #                     [
    #                         x not in name
    #                         for x in [
    #                             "visual_model",
    #                             "vision_tower",
    #                             "mm_projector",
    #                             "text_hidden_fcs",
    #                         ]
    #                     ]
    #                 )
    #                 and any([x in name for x in lora_target_modules])
    #             ):
    #                 lora_module_names.add(name)
    #         return sorted(list(lora_module_names))

    #     lora_alpha = args.lora_alpha
    #     lora_dropout = args.lora_dropout
    #     lora_target_modules = find_linear_layers(
    #         model, args.lora_target_modules.split(",")
    #     )
    #     lora_config = LoraConfig(
    #         r=lora_r,
    #         lora_alpha=lora_alpha,
    #         target_modules=lora_target_modules,
    #         lora_dropout=lora_dropout,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )
    #     model = get_peft_model(model, lora_config)
    #     model.print_trainable_parameters()

    # model.resize_token_embeddings(len(tokenizer))


    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    
    # 这个dataset完全可以不变，因为这里面就又refcoco等等数据集
    
    train_dataset = HybridDataset(
        args.dataset_dir,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        model_type = args.model_type,
    
    )

    if args.no_eval == False:
        val_dataset = ValDataset(
            args.dataset_dir,
            args.val_dataset,
            args.image_size,
            model_type = args.model_type,
            
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset 这里我们还是要改成是refcoco的val验证集
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                local_rank=args.local_rank,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        giou, ciou = validate(val_loader, model_engine, 0, writer, args)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        if not args.no_eval:
            giou, ciou = validate(val_loader, model_engine, epoch, writer, args)
            with open("rs.txt", "a") as f:
                f.write("Epoch {}: GIoU = {:.3f}, CIoU = {:.3f}\n".format(epoch, giou, ciou))
            is_best = ciou > best_score  # ciou as main metric
            best_score = max(ciou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            giou, cur_ciou  # Ensure both GIoU and CIoU are logged
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    # ce_losses = AverageMeter("CeLoss", ":.4f")s
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
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

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

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
            output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        print(intersection, union)
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

    return giou, ciou


if __name__ == "__main__":
    main(sys.argv[1:])