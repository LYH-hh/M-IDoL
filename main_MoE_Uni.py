# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from functools import partial

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models as torchvision_models


import utils
import MIDoL_model as vits
from MIDoL_model import MoEHead
from utils_moe import hook_scale_grad

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='swin_base', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0001, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-8, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=2, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default=[
                                                #'/lyh/dataset/baoshan_raw',
                                                #'/lyh/dataset/Fudus_Hospital/jiading/origin', 
                                                #'/lyh/dataset/Fudus_Hospital/English/origin',
                                                #'/lyh/dataset/fundus_dataset/AIROGS',
                                                '/lyh/dataset/fundus_dataset/EYEPACS/images',
                                                #'/lyh/dataset/fundus_dataset/APTOS2019',
                                                #'/lyh/dataset/wellPrepared/chest-ray',
                                                #'/lyh/dataset/fundus_dataset/Glaucoma_fundus',
                                                #'/lyh/dataset/fundus_dataset/IDRiD_data',
                                                #'/lyh/dataset/fundus_dataset/NEH_UT_2021RetinalOCTDataset',
                                                #'/lyh/dataset/fundus_dataset/OCT_TVHL',
                                                '/lyh/dataset/fundus_dataset/OCTc8',
                                                #'/lyh/dataset/fundus_dataset/OCTDL',
                                                #'/lyh/dataset/fundus_dataset/OCTDL',
                                                #'/lyh/dataset/fundus_dataset/OCTID',
                                                #'/lyh/dataset/fundus_dataset/PAPILA',
                                                #'/lyh/dataset/fundus_dataset/Retina',
                                                '/lyh/dataset/wellPrepared/OCT2017',
                                                # '/lyh/dataset/mimic_cxr/images', 
                                                # '/lyh/dataset/RSNA_CXR', 
                                                '/lyh/dataset/ChestXray8',
                                                # '/lyh/dataset/siim/images',
                                                # '/lyh/dataset/ChestXpertFrontal',
                                                # '/lyh/dataset/COVID-19_Radiography_Database',
                                                # '/lyh/dataset/Path_datasets/NCH_7K',
                                                '/lyh/dataset/Path_datasets/NCH-100k',
                                                # '/lyh/dataset/Path_datasets/BreakHis/dataset_cancer_v1',
                                                '/lyh/dataset/Path_datasets/MHIST',
                                                # '/lyh/dataset/Path_datasets/Mitosis_Detection/images',
                                                # '/lyh/dataset/Path_datasets/PanNuke/train/images',
                                                # '/lyh/dataset/Path_datasets/PanNuke/val',
                                                # '/lyh/dataset/Derm_datsets/ISIC_2024_Training_Input',
                                                # '/lyh/dataset/Derm_datsets/MILK10k_Test_Input',
                                                '/lyh/dataset/Derm_datsets/MILK10k_Training_Input',
                                                # '/lyh/dataset/Derm_datsets/ISIC_2016/3b',
                                                # '/lyh/dataset/Derm_datsets/ISIC2017',
                                                '/lyh/yzt/datasets/HAM10000',
                                                ], type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="./output/dino_swin_moe_0403", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=2, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser

class CustomImageDataset(Dataset):
    def __init__(self, folder_paths, transform=None):
        self.transform = transform
        self.image_paths = self._find_image_files_in_paths(folder_paths)  # 方法名更新
    
    def _find_image_files_in_paths(self, paths):
        """查找所有支持的图像文件"""
        files = []
        supported_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.TIF', '.TIFF'}
        
        for path in paths:
            if not os.path.exists(path):
                print(f"警告: 路径不存在 {path}")
                continue
                
            for dirpath, _, filenames in os.walk(path):
                for file in filenames:
                    file_ext = os.path.splitext(file)[1].lower()
                    if file_ext in supported_extensions:
                        files.append(os.path.join(dirpath, file))
        
        print(f"找到 {len(files)} 张图像")
        return files

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 使用PIL读取（简单但有限制）
        try:
            image = Image.open(img_path).convert("RGB")  # 强制转为RGB
        except Exception as e:
            print(f"无法读取 {img_path}: {e}")
            # 返回空白图像作为容错
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0

def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset = CustomImageDataset(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch]()
        teacher = vits.__dict__[args.arch]()
        embed_dim = student.num_features
    weights_dict = torch.load("/lyh/code/checkpoints/swin_base_patch4_window7_224_22k.pth", map_location=torch.device('cuda'))['model']

    # 创建目标模型的状态字典
    model_dict = student.state_dict()

    # 1. 首先处理不需要的head参数（可选）
    # for exit_name in list(weights_dict.keys()):
    #     if "head" in exit_name:
    #         del weights_dict[exit_name]

    # 2. 构建参数名称映射关系
    pretrained_dict = {}
    for k, v in weights_dict.items():
        # 将预训练权重键名改为目标模型的命名格式
        #new_k = 'module.backbone.' + k
        new_k = k
        
        # 只有当新键名存在于目标模型中时才保留
        if new_k in model_dict:
            pretrained_dict[new_k] = v
            print({new_k})
        else:
            print(f"Skipping parameter: {k} (not found in target model)")

    # 3. 更新模型参数
    model_dict.update(pretrained_dict)
    load_result = student.load_state_dict(model_dict, strict=False)

    # 4. 打印加载结果
    print("Missing keys:", load_result.missing_keys)
    print("Unexpected keys:", load_result.unexpected_keys)



    # For Tutel MoE
    for name, param in student.named_parameters():
        if param.requires_grad == True and hasattr(param, 'skip_allreduce') and param.skip_allreduce is True:
            student.add_param_to_skip_allreduce(name)
            param.register_hook(partial(hook_scale_grad, dist.get_world_size()))
            print(f"[rank{dist.get_rank()}] [{name}] skip all_reduce and div {dist.get_world_size()} for grad")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, MoEHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        MoEHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    n_parameters = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters / 1e6:.2f}M")
    flops = student.backbone.flops()
    print(f"Backbone FLOPs: {flops / 1e9:.2f}G")
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])

    
    # teacher and student start with the same weights
    print("___________Teacher______________")
    print(teacher_without_ddp.load_state_dict(student.module.state_dict()))
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        if epoch % 5 == 0 or epoch == args.epochs:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output, teacher_moe_logits = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output, student_moe_logits = student(images)
            loss, last_route, last_cst = dino_loss(student_output, teacher_output, teacher_moe_logits, student_moe_logits, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(route_loss=last_route)
        metric_logger.update(cst_loss=last_cst)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
        sinkhorn_eps=0.05,
        sinkhorn_iters=3,
        route_weight=1.0,
        cst_weight=1.0
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops

        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.route_weight = route_weight
        self.cst_weight = cst_weight

        self.register_buffer("center", torch.zeros(1, out_dim))

        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

        # optional logs
        self.last_dino = None
        self.last_route = None
        self.last_cst = None

    @staticmethod
    def _check_finite(name, x):
        if x is None:
            return
        if not torch.isfinite(x).all():
            bad = x[~torch.isfinite(x)]
            raise RuntimeError(
                f"[NaN/Inf] {name} shape={tuple(x.shape)} dtype={x.dtype} "
                f"min={x.min().item()} max={x.max().item()} "
                f"bad_count={bad.numel()} bad_example={bad.flatten()[:5].tolist()}"
            )

    def forward(self, student_output, teacher_output, teacher_moe_logits, student_moe_logits, epoch):
        """
          teacher_output      : [B*T, D] 
          teacher_moe_logits  : [B*T, N] 
          student_output      : [B*M, D] 
          student_moe_logits  : [B*M, N] 
        """
        self._check_finite("teacher_output(in)", teacher_output)
        self._check_finite("student_output(in)", student_output)
        self._check_finite("teacher_moe_logits(in)", teacher_moe_logits)
        self._check_finite("student_moe_logits(in)", student_moe_logits)

        M = self.ncrops
        assert student_output.shape[0] % M == 0, "student_output first dim must be divisible by ncrops"
        B = student_output.shape[0] // M

        assert teacher_output.shape[0] % B == 0, "teacher_output first dim must be divisible by inferred batch size B"
        T = teacher_output.shape[0] // B

        student_out = (student_output / self.student_temp).chunk(M)  # M * [B, D]

        temp = float(self.teacher_temp_schedule[epoch])
        teacher_prob = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_prob = teacher_prob.detach().chunk(T)  # T * [B, D]

        cst_loss = 0.0
        n_terms = 0
        for iq, q in enumerate(teacher_prob):
            for v in range(len(student_out)):
                # skip same-view only when student view index aligns with teacher view index
                if v < T and v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                cst_loss += loss.mean()
                n_terms += 1
        cst_loss = cst_loss / max(n_terms, 1)
        self.update_center(teacher_output)

        route_loss = self._routing_consistency_loss(
            student_moe_logits=student_moe_logits,
            teacher_moe_logits=teacher_moe_logits,
            B=B, M=M, T=T, temp=temp
        )

        total = self.route_weight * route_loss + self.cst_weight * cst_loss

        self.last_route = float(route_loss.detach().cpu())
        self.last_cst = float(cst_loss.detach().cpu())
        return total, self.last_route, self.last_cst

    # ----------------------
    # Sinkhorn-Knopp (teacher router)
    # ----------------------
    @torch.no_grad()
    def _sinkhorn_knopp(self, logits):
        """
        Stabilized Sinkhorn-Knopp (SwAV-style), distributed-aware.

        logits: [B, N]
        return: [B, N]  (each row sums to 1)
        """
        eps = float(self.sinkhorn_eps)
        iters = int(self.sinkhorn_iters)
        tiny = 1e-12

        # ---- FP32 for stability ----
        x = logits.float()

        # ---- max-shift to avoid exp overflow ----
        x = x - x.max(dim=1, keepdim=True).values  # [B, N]

        # Q: [N, B]
        Q = torch.exp(x / eps).t()
        Q = torch.clamp(Q, min=tiny)
        Q = Q / (Q.sum() + tiny)

        world_size = 1
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()

        K, B_local = Q.shape
        B_global = B_local * world_size

        for _ in range(iters):
            # Row normalize: each row -> 1/K
            sum_rows = Q.sum(dim=1, keepdim=True)  # [K,1]
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(sum_rows)
            Q = Q / (sum_rows + tiny)
            Q = Q / K

            # Col normalize: each col -> 1/B_global
            sum_cols = Q.sum(dim=0, keepdim=True)  # [1,B_local]
            Q = Q / (sum_cols + tiny)
            Q = Q / B_global

            Q = torch.clamp(Q, min=tiny)

        # make columns sum to 1 (per local batch)
        Q = Q * B_global  # now each column sums to ~1

        out = Q.t().contiguous()  # [B_local, K] == [B, N]
        # ensure each row sums to 1 exactly (safe)
        out = torch.clamp(out, min=tiny)
        out = out / (out.sum(dim=1, keepdim=True) + tiny)
        return out


    def _routing_consistency_loss(self, student_moe_logits, teacher_moe_logits, B, M, T, temp):
        """
        Correct & stable implementation of Eq.(gateloss) minimization form:

        L_route = - avg_{j,g}  E_b [ sum_i a_S^{(j,b,i)} * log a_T^{(g,b,i)} ]

        student: a_S = softmax(student_logits / student_temp)
        teacher: a_T = sinkhorn(teacher_logits / temp)  (stop-grad)

        Shapes:
        student_moe_logits: [B*M, N]
        teacher_moe_logits: [B*T, N]
        """
        tiny = 1e-12

        # ---- student routing probs ----
        a_s_chunks = F.softmax(student_moe_logits / self.student_temp, dim=-1).chunk(M)  # M * [B, N]

        # ---- teacher routing probs (sinkhorn) ----
        with torch.no_grad():
            t_chunks = (teacher_moe_logits / temp).chunk(T)  # T * [B, N]
            a_t_chunks = []
            for x in t_chunks:
                q = self._sinkhorn_knopp(x)                  # [B, N]
                q = torch.clamp(q, min=tiny)                 # avoid log(0)
                a_t_chunks.append(q.detach())

        # ---- cross-view routing consistency ----
        loss = 0.0
        n_terms = 0
        for j in range(M):
            for g in range(T):
                # skip aligned views if student first T views align teacher's T views
                if j < T and j == g:
                    continue
                # CE(a_s || a_t) = - sum a_s * log(a_t)
                ce = -torch.sum(a_s_chunks[j] * torch.log(a_t_chunks[g]), dim=-1)  # [B]
                loss += ce.mean()
                n_terms += 1

        if n_terms == 0:
            return student_moe_logits.new_tensor(0.0)
        return loss / n_terms

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)

        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
    



class DINOLoss1(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, teacher_moe_logits, student_moe_logits, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)

#CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29503 main_MoE_Uni.py
#CUDA_VISIBLE_DEVICES=4 python main.py