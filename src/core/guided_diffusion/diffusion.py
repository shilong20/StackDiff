"""
【作用概述】封装扩散模型采样、材料分解与滑窗推理逻辑；核心输入来自 YAML 配置中的模型权重、输入图像目录与采样参数，输出为分解/采样结果图像及可选调试记录。
【关联说明】文件/模块：src/main.py（分解入口）；configs/separate/*.yml（推理配置）；src/core/guided_diffusion/script_util.py（模型构建）；src/core/functions/svd_operators.py（DDNM/SVD 相关算子）。
【命令行用法】本文件不直接作为脚本运行；请通过 `python src/main.py --config configs/separate/ReS2.yml` 间接调用。
"""

import os
import logging
import time
import glob
import math
import csv

import numpy as np
import tqdm
import torch
import torch.utils.data as data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion

import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

from scipy.linalg import orth

import torchvision.utils as vutils
from torchvision import transforms
from datasets.sorted_image_folder import SortedImageFolder


def _normalize_identifier(identifier: str) -> str:
    return identifier.replace("\\", "/")


def _resolve_plan(plans, identifier: str):
    if not plans:
        return None
    candidates = []
    normalized = _normalize_identifier(identifier)
    candidates.extend([identifier, normalized])
    rel_no_ext = os.path.splitext(normalized)[0]
    candidates.append(rel_no_ext)
    base_name = os.path.basename(normalized)
    candidates.append(base_name)
    candidates.append(os.path.splitext(base_name)[0])

    seen = set()
    for key in candidates:
        if not key or key in seen:
            continue
        seen.add(key)
        if key in plans:
            return plans[key]
    return None


def _split_identifier(identifier: str):
    normalized = _normalize_identifier(identifier)
    rel_no_ext = os.path.splitext(normalized)[0]
    rel_dir = os.path.dirname(rel_no_ext)
    base_name = os.path.basename(rel_no_ext)
    return normalized, rel_no_ext, rel_dir, base_name


def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)    

def separate2single(x,N,method='sum'):
    # Only sum method is supported
    x_gray = x.sum(dim=1, keepdim=True)[0] + (N - 1.0)
    x_repeated = x_gray.repeat(1, N, 1, 1)
    return x_repeated

def single2separate(x,N,method='sum'):
    # Only sum method is supported
    x_gray = x[:,0,:,:]
    x_gray = x_gray / N
    result =  torch.stack([x_gray] * N, dim=1)
    return result

def adaptive_separate2single(x, N, y_input_patch):
    """
    自适应叠加函数：I_sup = k * (I_1 + I_2 + ... + I_n)
    其中 k* = <y, ΣL_i> / <ΣL_i, ΣL_i>，最终取 k = min(k*, 1)
    """
    sum_result = x.sum(dim=1, keepdim=True)
    # 内积形式：<y, ΣL_i> / <ΣL_i, ΣL_i>
    numerator = (y_input_patch * sum_result).sum()
    denominator = (sum_result * sum_result).sum()
    denominator = torch.clamp(denominator, min=1e-8)
    k_star = numerator / denominator
    # 保证 k 不超过 1
    k = torch.clamp(k_star, max=1.0)
    x_gray = k * sum_result  # 移除(N-1.0)偏移，因为这是直接叠加
    x_repeated = x_gray.repeat(1, N, 1, 1)
    return x_repeated

def adaptive_single2separate(x, N, y_input_patch):
    """
    自适应反向分离函数
    """
    # 对于反向操作，我们不直接使用k，因为分离是线性的
    x_gray = x[:,0,:,:] / N
    result = torch.stack([x_gray] * N, dim=1)
    return result


def resolve_superposition_k(sum_result: torch.Tensor, target: torch.Tensor, args):
    """
    根据配置确定叠加系数 k：
    - 若指定 fixed_superposition_k，则全程使用该常数（会截断到 [0,1] 区间）
    - 否则在开启 adaptive_superposition 时按内积自适应求解
    - 默认返回 1.0（保持向后兼容）
    """
    fixed_k = getattr(args, "fixed_superposition_k", None)
    adaptive_enabled = getattr(args, "adaptive_superposition", False)

    if fixed_k is not None:
        k_tensor = sum_result.new_tensor(float(fixed_k))
        mode = "fixed"
    elif adaptive_enabled:
        numerator = (target * sum_result).sum()
        denominator = (sum_result * sum_result).sum()
        denominator = torch.clamp(denominator, min=1e-8)
        k_star = numerator / denominator
        k_tensor = torch.clamp(k_star, min=0.0, max=1.0)
        mode = "adaptive"
    else:
        k_tensor = sum_result.new_tensor(1.0)
        mode = "unity"

    # 避免数值为负或过小
    k_tensor = torch.clamp(k_tensor, min=0.0, max=1.0)
    k_safe = torch.clamp(k_tensor, min=1e-8)
    return k_tensor, k_safe, mode


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self, simplified):
        cls_fn = None
        if self.config.model.type == 'simple':
            model = Model(self.config)

            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                # print("Loading checkpoint {}".format(ckpt))  # 禁用日志输出
            elif name == 'celeba_hq':
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt',
                             ckpt)
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'STEM_clean':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict) # 返回一个unet
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            # ckpt = os.path.join(self.args.exp, "logs/mdsave/ema_0.9999_380000.pt")
            ckpt = os.path.join(self.args.exp, "logs/part1_plus_mask_plus_full_mask_mdsave/ema_0.9999_260000.pt")
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)
        
        elif self.config.model.type == 'STEM_separate':
            # 提取create_model需要的参数，移除不需要的参数
            base_model_args = {
                'image_size': self.config.model.image_size,
                'num_channels': self.config.model.num_channels,
                'num_res_blocks': self.config.model.num_res_blocks,
                'learn_sigma': self.config.model.learn_sigma,
                'class_cond': self.config.model.class_cond,
                'attention_resolutions': self.config.model.attention_resolutions,
                'num_heads': self.config.model.num_heads,
                'num_head_channels': self.config.model.num_head_channels,
                'num_heads_upsample': self.config.model.num_heads_upsample,
                'use_scale_shift_norm': self.config.model.use_scale_shift_norm,
                'dropout': self.config.model.dropout,
                'resblock_updown': self.config.model.resblock_updown,
                'use_fp16': self.config.model.use_fp16,
                'use_new_attention_order': self.config.model.use_new_attention_order,
            }

            def load_one_model(ckpt_path: str):
                m = create_model(**base_model_args)
                if self.config.model.use_fp16:
                    m.convert_to_fp16()
                m.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                m.to(self.device)
                m.eval()
                return torch.nn.DataParallel(m)

            self.model_list = []
            ckpt_list = []
            if hasattr(self.config, 'paths'):
                if hasattr(self.config.paths, 'model_checkpoints') and isinstance(self.config.paths.model_checkpoints, (list, tuple)):
                    ckpt_list = list(self.config.paths.model_checkpoints)
                elif hasattr(self.config.paths, 'model_checkpoint'):
                    ckpt_list = [self.config.paths.model_checkpoint]

            if not ckpt_list:
                raise ValueError("STEM_separate requires paths.model_checkpoint or paths.model_checkpoints in the config.")

            for ck in ckpt_list:
                self.model_list.append(load_one_model(ck))

            # 兼容旧逻辑传入的model参数，取首个作为代表
            model = self.model_list[0]

        elif self.config.model.type == 'STEM_separate_mixed':
            # 混合材料的处理，与STEM_separate相同的逻辑
            # 提取create_model需要的参数，移除不需要的参数
            base_model_args = {
                'image_size': self.config.model.image_size,
                'num_channels': self.config.model.num_channels,
                'num_res_blocks': self.config.model.num_res_blocks,
                'learn_sigma': self.config.model.learn_sigma,
                'class_cond': self.config.model.class_cond,
                'attention_resolutions': self.config.model.attention_resolutions,
                'num_heads': self.config.model.num_heads,
                'num_head_channels': self.config.model.num_head_channels,
                'num_heads_upsample': self.config.model.num_heads_upsample,
                'use_scale_shift_norm': self.config.model.use_scale_shift_norm,
                'dropout': self.config.model.dropout,
                'resblock_updown': self.config.model.resblock_updown,
                'use_fp16': self.config.model.use_fp16,
                'use_new_attention_order': self.config.model.use_new_attention_order,
            }

            def load_one_model(ckpt_path: str):
                m = create_model(**base_model_args)
                if self.config.model.use_fp16:
                    m.convert_to_fp16()
                m.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                m.to(self.device)
                m.eval()
                return torch.nn.DataParallel(m)

            self.model_list = []
            ckpt_list = []
            if hasattr(self.config, 'paths'):
                if hasattr(self.config.paths, 'model_checkpoints') and isinstance(self.config.paths.model_checkpoints, (list, tuple)):
                    ckpt_list = list(self.config.paths.model_checkpoints)
                elif hasattr(self.config.paths, 'model_checkpoint'):
                    ckpt_list = [self.config.paths.model_checkpoint]

            if not ckpt_list:
                raise ValueError("STEM_separate_mixed requires paths.model_checkpoint or paths.model_checkpoints in the config.")

            for ck in ckpt_list:
                self.model_list.append(load_one_model(ck))

            # 兼容旧逻辑传入的model参数，取首个作为代表
            model = self.model_list[0]
        
        else:
            raise ValueError(f"Unsupported model type: {self.config.model.type}")

        if simplified:
            # print('Run Simplified DDNM, without SVD.',
            #       f'{self.config.time_travel.T_sampling} sampling steps.',
            #       f'travel_length = {self.config.time_travel.travel_length},',
            #       f'travel_repeat = {self.config.time_travel.travel_repeat}.',
            #       f'Task: {self.args.deg}.'
            #      )  # 禁用日志输出
            self.simplified_ddnm_plus(model, cls_fn)
        else:
            # print('Run SVD-based DDNM.',
            #       f'{self.config.time_travel.T_sampling} sampling steps.',
            #       f'travel_length = {self.config.time_travel.travel_length},',
            #       f'travel_repeat = {self.config.time_travel.travel_repeat}.',
            #       f'Task: {self.args.deg}.'
            #      )  # 禁用日志输出
            self.svd_based_ddnm_plus(model, cls_fn)
            
            
    def simplified_ddnm_plus(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        # print(f'Dataset has size {len(test_dataset)}')  # 禁用日志输出
        # print(f'Dataset has size {len(test_dataset)}');  // 禁用日志输出

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        # 受限环境下禁用多进程加载，避免 IPC 权限问题
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation operator
        # print("args.deg:",args.deg)  # 禁用日志输出
        if args.deg =='colorization':
            A = lambda z: color2gray(z)
            Ap = lambda z: gray2color(z)
        elif args.deg =='separate_sum':
            A = lambda z: separate2single(z,args.N,method='sum')
            Ap = lambda z: single2separate(z,args.N,method='sum')
        
        elif args.deg =='denoising':
            A = lambda z: z
            Ap = A

        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        sigma_y = args.sigma_y
        
        # print(f'Start from {args.subset_start}')  # 禁用日志输出
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        # call = 0
        for y, classes, filename_batch in pbar:
            identifier = filename_batch[0]
            file_identifier, _, identifier_dir, identifier_base = _split_identifier(identifier)

            output_root = self.args.image_folder
            if identifier_dir:
                output_root = os.path.join(output_root, identifier_dir)
            image_folder = os.path.join(output_root, identifier_base)
            os.makedirs(image_folder, exist_ok=True)

            # 断点重续：检查所有输出文件是否已存在
            all_outputs_exist = True
            expected_outputs = [
                os.path.join(image_folder, f"{identifier_base}_original.png")
            ]
            for i in range(args.N):
                expected_outputs.append(os.path.join(image_folder, f"{identifier_base}_{i}.png"))
            if args.deg == 'separate_sum':
                expected_outputs.append(os.path.join(image_folder, f"{identifier_base}_conbine.png"))
            
            for output_file in expected_outputs:
                if not os.path.exists(output_file):
                    all_outputs_exist = False
                    break
            
            if all_outputs_exist:
                pbar.set_description(f"Skipping (already processed): {identifier_base}")
                continue

            debug_residual_enabled = getattr(self.args, 'residual_debug', False)
            debug_x0_dir = None
            debug_x0_counter = 0
            if debug_residual_enabled:
                debug_x0_dir = os.path.join(image_folder, "debug", "x0_t")
                os.makedirs(debug_x0_dir, exist_ok=True)

            y = y.to(self.device)

            y = data_transform(self.config, y)

            if config.sampling.batch_size!=1:
                raise ValueError("please change the config file to set batch size as 1")

###################################################

            # init x_T
            plans = getattr(args, 'image_crop_plans', {}) or {}
            plan = _resolve_plan(plans, file_identifier)
            row_use = getattr(args, 'row', 1)
            col_use = getattr(args, 'col', 1)
            if plan is not None:
                row_use = getattr(plan, 'row', row_use) or row_use
                col_use = getattr(plan, 'col', col_use) or col_use
            x = torch.randn(
                y.shape[0],
                args.N * config.data.channels,
                row_use*config.data.image_size,
                col_use*config.data.image_size,
                device=self.device,
            )

            residual_debug_records = []
            residual_debug_counter = 0

            with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]
                
                times = get_schedule_jump(config.time_travel.T_sampling, 
                                               config.time_travel.travel_length, 
                                               config.time_travel.travel_repeat,
                                              )
                time_pairs = list(zip(times[:-1], times[1:]))
                  
                # prepare for shift
                H_target = x.size(2)
                W_target = x.size(3)

                finalresult = torch.zeros_like(x)

                tile_px = config.data.image_size
                stride_px = getattr(args, 'window_stride_px', tile_px // 2)
                if stride_px is None:
                    stride_px = tile_px // 2
                stride_px = int(stride_px)
                stride_px = max(1, min(stride_px, tile_px))
                max_h_start = max(0, H_target - tile_px)
                max_w_start = max(0, W_target - tile_px)
                shift_h_total = max(1, math.floor(max_h_start / stride_px) + 1)
                shift_w_total = max(1, math.floor(max_w_start / stride_px) + 1)
                overlap_px = max(0, tile_px - stride_px)
                total_shifts = shift_h_total * shift_w_total

                with tqdm.tqdm(total=total_shifts) as pbar:
                    pbar.set_description('total shifts')
    
                    # shift along H
                    for shift_h in range(shift_h_total):
                        h_l = min(stride_px * shift_h, max_h_start)
                        h_r = h_l + tile_px

                        # shift along W
                        for shift_w in range(shift_w_total):
                            x_temp=finalresult
                            w_l = min(stride_px * shift_w, max_w_start)
                            w_r = w_l + tile_px
                            
                            # 范围已确定

                            # 调整数据大小以适应原模型
                            x0_preds = []
                            x0_t_hats = []
                            xs = [x[:, :, h_l:h_r, w_l:w_r]]
                            y_patch = y[:, :, h_l:h_r, w_l:w_r]

                            # reverse diffusion sampling
                            for i, j in tqdm.tqdm(time_pairs):
                                i, j = i*skip, j*skip
                                if j<0: j=-1 

                                if j < i: # normal sampling 
                                    t = (torch.ones(n) * i).to(x.device)
                                    # print(f't: {t}')
                                    next_t = (torch.ones(n) * j).to(x.device)
                                    at = compute_alpha(self.betas, t.long())
                                    at_next = compute_alpha(self.betas, next_t.long())
                                    sigma_t = (1 - at_next**2).sqrt()

                                    # 每个通道单独计算
                                    et_list = []
                                    for no in range(args.N):
                                        xt = xs[-1][:,no,:,:].unsqueeze(1).to('cuda')

                                        # # save xt[0][0]
                                        # xt_img = inverse_data_transform(config, xt[0][0])
                                        # tvu.save_image(
                                        #     xt_img, os.path.join(self.args.image_folder, f"{filename}_{no}_{int(t)}.png")
                                        # )

                                        # 若提供了多模型权重，则按层选择对应模型；否则使用单一模型
                                        model_to_use = model
                                        if hasattr(self, 'model_list') and len(self.model_list) > 1:
                                            idx = min(no, len(self.model_list)-1)
                                            model_to_use = self.model_list[idx]
                                        et = model_to_use(xt, t)
                                        if et.size(1) == 2:
                                            et = et[:, :1]
                                            et_list.append(et)

                                    et = torch.cat(et_list, dim=1)
                                    xt = xs[-1].to('cuda')

                                    ###############
                                    # Eq. 12
                                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                                    xt_debug = None
                                    if debug_x0_dir is not None:
                                        xt_debug = inverse_data_transform(config, xt.detach().clone())
                                        xt_debug = xt_debug.to('cpu')
                                        layer_count = min(args.N, xt_debug.size(1))
                                        for layer_idx in range(layer_count):
                                            layer_tensor = xt_debug[0][layer_idx]
                                            debug_name = f"{identifier_base}_iter{debug_x0_counter:05d}_layer{layer_idx:02d}.png"
                                            tvu.save_image(
                                                layer_tensor,
                                                os.path.join(debug_x0_dir, debug_name),
                                            )
                                        debug_x0_counter += 1

                                    # Eq. 19
                                    if sigma_t >= at_next*sigma_y:
                                        lambda_t = 1.
                                        gamma_t = (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
                                    else:
                                        lambda_t = (sigma_t)/(at_next*sigma_y)
                                        gamma_t = 0.

                                    # Eq. 17
                                    if self.args.deg == 'separate_sum':
                                        # 在 [0,1] 域上定义 A/Ap，形式保持一致
                                        L = (x0_t + 1.0) / 2.0
                                        y_pos = (y_patch + 1.0) / 2.0
                                        sum_result = L.sum(dim=1, keepdim=True)
                                        k_tensor, k_safe, k_mode = resolve_superposition_k(sum_result, y_pos, self.args)
                                        A_pos_L = k_tensor * sum_result
                                        k_value = float(k_tensor.item())

                                        inv_factor = (1.0 / args.N) / k_safe
                                        Ap_pos = lambda residual, inv_factor=inv_factor: (
                                            residual * inv_factor
                                        ).repeat(1, args.N, 1, 1)
                                        residual_pos = A_pos_L - y_pos
                                        L_hat = L - lambda_t * Ap_pos(residual_pos)
                                        x0_t_hat = 2.0 * L_hat - 1.0

                                        if debug_residual_enabled:
                                            residual_abs_mean = float(residual_pos.abs().mean().item())
                                            residual_debug_records.append({
                                                "iter": residual_debug_counter,
                                                "shift_h": int(shift_h),
                                                "shift_w": int(shift_w),
                                                "shift_index": int(shift_h * shift_w_total + shift_w),
                                                "t": int(i),
                                                "t_next": int(j),
                                                "lambda_t": float(lambda_t),
                                                "k": k_value,
                                                "residual_abs_mean": residual_abs_mean,
                                                "adaptive": bool(k_mode == "adaptive"),
                                                "k_mode": k_mode,
                                            })

                                            # 使用当前的 k 对 xt 做自适应叠加，并在 debug 目录下额外保存一张 xt 叠加图
                                            if debug_x0_dir is not None and xt_debug is not None:
                                                xt_layers = xt_debug  # [B, N, H, W]，每个通道对应一层 xt
                                                xt_sum = xt_layers.sum(dim=1, keepdim=True)
                                                # 保证 k 与 xt 在同一 device 上
                                                k_for_xt = k_tensor.to(xt_sum.device)
                                                xt_combine = torch.clamp(k_for_xt * xt_sum, 0.0, 1.0)
                                                xt_combine = xt_combine[0]  # [C, H, W]
                                                xt_combine_name = f"{identifier_base}_iter{residual_debug_counter:05d}_xt_combine.png"
                                                tvu.save_image(
                                                    xt_combine,
                                                    os.path.join(debug_x0_dir, xt_combine_name),
                                                )

                                            residual_debug_counter += 1
                                    else:
                                        x0_t_hat = x0_t - lambda_t*Ap(A(x0_t) - y_patch)
                                    x0_t_hats.append(x0_t_hat.to('cpu'))
                                    # mask-shift trick 
                                    if overlap_px > 0:
                                        if shift_w == 0 and shift_h != 0:
                                            neo_h_l = h_l
                                            neo_h_r = neo_h_l + overlap_px
                                            x0_t_hat[:, :, 0:overlap_px, :] = x_temp[:, :, neo_h_l:neo_h_r, w_l:w_r].to('cuda')
                                        elif shift_w != 0:
                                            neo_w_l = w_l
                                            neo_w_r = neo_w_l + overlap_px
                                            neo_h_l = h_l
                                            neo_h_r = h_l + tile_px
                                            x0_t_hat[:, :, :, 0:overlap_px] = x_temp[:, :, neo_h_l:neo_h_r, neo_w_l:neo_w_r].to('cuda')
                                            if shift_h != 0:
                                                neo_h_l_top = h_l
                                                neo_h_r_top = neo_h_l_top + overlap_px
                                                neo_w_l_top = w_l
                                                neo_w_r_top = w_l + tile_px
                                                x0_t_hat[:, :, 0:overlap_px, :] = x_temp[:, :, neo_h_l_top:neo_h_r_top, neo_w_l_top:neo_w_r_top].to('cuda')

                                    eta = self.args.eta

                                    c1 = (1 - at_next).sqrt() * eta
                                    c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                                    # different from the paper, we use DDIM here instead of DDPM
                                    xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

                                    x0_preds.append(x0_t.to('cpu'))
                                   
                                    xs.append(xt_next.to('cpu'))    

                                else: # time-travel back
                                    next_t = (torch.ones(n) * j).to(x.device)
                                    at_next = compute_alpha(self.betas, next_t.long())
                                    x0_t = x0_preds[-1].to('cuda')

                                    xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                                    xs.append(xt_next.to('cpu'))
                            # print(f'len(xs): {len(xs)}')
                            finalresult[:,:,h_l:h_r,w_l:w_r] = xs[-1].to(finalresult.device)
                            pbar.update(1)
                            # print(f'finalresult.shape: {finalresult.shape}')
                x = [finalresult]
                # step = 0
                # x0_preds = [x0_preds[step]]
                # x0_t_hats = [x0_t_hats[step]]
            x = [inverse_data_transform(config, xi) for xi in x]
            # x0_preds = [inverse_data_transform(config, xi) for xi in x0_preds]
            # x0_t_hats = [inverse_data_transform(config, xi) for xi in x0_t_hats]

            if debug_residual_enabled and residual_debug_records:
                debug_path = os.path.join(image_folder, f"{identifier_base}_residual_debug.csv")
                fieldnames = [
                    "iter",
                    "shift_h",
                    "shift_w",
                    "shift_index",
                    "t",
                    "t_next",
                    "lambda_t",
                    "k",
                    "k_mode",
                    "residual_abs_mean",
                    "adaptive",
                ]
                with open(debug_path, "w", newline="") as debug_file:
                    writer = csv.DictWriter(debug_file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(residual_debug_records)
                try:
                    iters = [record["iter"] for record in residual_debug_records]
                    k_values = [record["k"] for record in residual_debug_records]
                    residual_means = [record["residual_abs_mean"] for record in residual_debug_records]
                    fig, ax1 = plt.subplots(figsize=(8, 4))
                    ax1.plot(iters, k_values, color="tab:blue", label="k")
                    ax1.set_xlabel("iter")
                    ax1.set_ylabel("k", color="tab:blue")
                    ax1.tick_params(axis="y", labelcolor="tab:blue")
                    ax2 = ax1.twinx()
                    ax2.plot(iters, residual_means, color="tab:orange", label="residual_abs_mean")
                    ax2.set_ylabel("|residual_pos| mean", color="tab:orange")
                    ax2.tick_params(axis="y", labelcolor="tab:orange")
                    fig.tight_layout()
                    plot_path = os.path.join(image_folder, f"{identifier_base}_residual_debug.png")
                    fig.savefig(plot_path, dpi=200)
                    plt.close(fig)
                except Exception as exc:
                    logging.warning(f"residual debug plot failed: {exc}")
            
            # 保存原图到输出文件夹
            original_y = inverse_data_transform(config, y)
            tvu.save_image(
                original_y[0], os.path.join(image_folder, f"{identifier_base}_original.png")
            )
            
            # 保存分解后的层图像
            for i in range (args.N):
                tvu.save_image(
                x[0][0][i], os.path.join(image_folder, f"{identifier_base}_{i}.png")
                )
        
                # tvu.save_image(
                # x0_preds[0][0][i], os.path.join(self.args.image_folder, f"{identifier_base}_{i}_x0_preds.png")
                # )
                # tvu.save_image(
                # x0_t_hats[0][0][i], os.path.join(self.args.image_folder, f"{identifier_base}_{i}_x0_t_hats.png")
                # )
                # print(f'x0_preds[0]: {x0_preds[0].shape}')
                # print(f'x0_t_hats[0]: {x0_t_hats[0].shape}')

            if args.deg == 'separate_sum':
                layers = x[0][0]
                sum_layers = layers.sum(dim=0, keepdim=True)
                y_pos = original_y[:1]
                k_tensor, _, _ = resolve_superposition_k(sum_layers, y_pos, self.args)
                conbine = torch.clamp(k_tensor * sum_layers, 0.0, 1.0)
                conbine = conbine.squeeze(0).to('cuda')

                tvu.save_image(
                        conbine, os.path.join(image_folder, f"{identifier_base}_conbine.png")
                        )

    def svd_based_ddnm_plus(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)
         
        # dataset = SortedImageFolder(
        #     os.path.join(args.exp, "datasets", "stem"),
        #     transform=transforms.Compose([
        #         transforms.Resize([args.n_unit * config.data.image_size, args.n_unit * config.data.image_size]),
        #         transforms.Grayscale(num_output_channels=1),
        #         transforms.ToTensor()
        #     ]),
        # sort=False)  

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            dataset = torch.utils.data.Subset(dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(dataset)

        # print(f'Dataset has size {len(dataset)}')  # 禁用日志输出


        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
 
        val_loader = data.DataLoader(
            dataset,
            batch_size=config.sampling.batch_size,
            shuffle=False,
            num_workers=1,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation matrix
        deg = args.deg
        A_funcs = None
        if deg == 'cs_walshhadamard':
            compress_by = round(1/args.deg_scale)
            from functions.svd_operators import WalshHadamardCS
            A_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by,
                                      torch.randperm(self.config.data.image_size ** 2, device=self.device), self.device)
        elif deg == 'cs_blockbased':
            cs_ratio = args.deg_scale
            from functions.svd_operators import CS
            A_funcs = CS(config.data.channels, self.config.data.image_size, cs_ratio, self.device)
        elif deg == 'inpainting':
            from functions.svd_operators import Inpainting
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            A_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
        elif deg == 'denoising':
            from functions.svd_operators import Denoising
            A_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'colorization':
            from functions.svd_operators import Colorization
            A_funcs = Colorization(config.data.image_size, self.device)
        elif deg == 'sr_averagepooling':
            blur_by = int(args.deg_scale)
            from functions.svd_operators import SuperResolution
            A_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
        elif deg == 'sr_bicubic':
            factor = int(args.deg_scale)
            from functions.svd_operators import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            A_funcs = SRConv(kernel / kernel.sum(), \
                             config.data.channels, self.config.data.image_size, self.device, stride=factor)
        elif deg == 'deblur_uni':
            from functions.svd_operators import Deblurring
            A_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(self.device), config.data.channels,
                                 self.config.data.image_size, self.device)
        elif deg == 'deblur_gauss':
            from functions.svd_operators import Deblurring
            sigma = int(args.deblur_sigma)
            # sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            A_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_aniso':
            from functions.svd_operators import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            A_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels,
                                   self.config.data.image_size, self.device)
        else:
            raise ValueError("degradation type not supported")
        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        sigma_y = args.sigma_y
        
        # print(f'Start from {args.subset_start}')  # 禁用日志输出
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes, filename_batch in pbar:
            identifier = filename_batch[0]
            _, _, identifier_dir, identifier_base = _split_identifier(identifier)
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            #Start DDIM
            x = torch.randn(
                x_orig.shape[0],
                config.data.channels,
                args.row * config.data.image_size,
                args.col * config.data.image_size,
                device=self.device,
            )

            batch_size = x.shape[0]

            with torch.no_grad():
                if sigma_y==0.: # noise-free case, turn to ddnm
                    x, _ = ddnm_diffusion(x, model, self.betas, self.args.eta, A_funcs, y, cls_fn=cls_fn, classes=classes, config=config)
                else: # noisy case, turn to ddnm+
                    # x, _ = ddnm_plus_diffusion(x, model, self.betas, self.args.eta, A_funcs, y, sigma_y, cls_fn=cls_fn, classes=classes, config=config)
                    x = ddnm_plus_diffusion(x, model, self.betas, self.args.eta, A_funcs, x_orig, sigma_y, cls_fn=cls_fn, classes=classes, config=config)
                    
            x = [inverse_data_transform(config, xi) for xi in x]


            for j in range(x[0].size(0)):
                output_root = self.args.image_folder
                if identifier_dir:
                    output_root = os.path.join(output_root, identifier_dir)
                os.makedirs(output_root, exist_ok=True)
                tvu.save_image(
                    x[0][j], os.path.join(output_root, f"{identifier_base}.png")
                )


            idx_so_far += batch_size

        # print("Number of samples: %d" % (idx_so_far - idx_init))  # 禁用日志输出

# Code form RePaint   
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
