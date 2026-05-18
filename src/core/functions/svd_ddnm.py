import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os
import math

class_num = 951


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)

def ddnm_diffusion(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None):
    with torch.no_grad():

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat,
                              )
        time_pairs = list(zip(times[:-1], times[1:]))
        
        # reverse diffusion sampling
        for i, j in tqdm(time_pairs):
            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 2:
                    et = et[:, :1]

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                x0_t_hat = x0_t - A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(*x0_t.size())

                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]


def ddnm_plus_diffusion(x, model, b, eta, A_funcs,x_orig, sigma_y, cls_fn=None, classes=None, config=None):
    with torch.no_grad():

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        # x0_preds = []
        # xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat,
                              )
        time_pairs = list(zip(times[:-1], times[1:])) 

        # prepare for shift
        H_target = x.size(2)
        W_target = x.size(3)

        finalresult = torch.zeros_like(x)

        shift_h_total = math.ceil(H_target/64)-1
        shift_w_total = math.ceil(W_target/64)-1

        with tqdm(total=shift_h_total*shift_w_total) as pbar:
            pbar.set_description('total shifts')
            
            # shift along H
            for shift_h in range(shift_h_total):
                h_l = int(64*shift_h)
                h_r = h_l+128

                # shift along W
                for shift_w in range(shift_w_total):
                    x_temp=finalresult
                    w_l = int(64*shift_w)
                    w_r = w_l+128
                
                    # 范围已确定

                    # 调整数据大小以适应原模型
                    x0_preds = []
                    xs = [x[:, :, h_l:h_r, w_l:w_r]]
                    x_orig_patch = x_orig[:, :, h_l:h_r, w_l:w_r]

                    y_patch = A_funcs.A(x_orig_patch)

                    batch, hwc = y_patch.size()
                    hw = hwc
                    h = w = int(hw ** 0.5)
                    y_patch = y_patch.reshape((batch, 1, h, w))
                    y_patch = y_patch.reshape((batch, hwc))

                    # reverse diffusion sampling
                    for i, j in tqdm(time_pairs):
                        i, j = i*skip, j*skip
                        if j<0: j=-1 

                        if j < i: # normal sampling 
                            t = (torch.ones(n) * i).to(x.device)
                            next_t = (torch.ones(n) * j).to(x.device)
                            at = compute_alpha(b, t.long())
                            at_next = compute_alpha(b, next_t.long())
                            xt = xs[-1].to('cuda')

                            et = model(xt, t)

                            if et.size(1) == 2:
                                et = et[:, :1]
                            et = et[:, 0:1, :, :]

                            # Eq. 12
                            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                            sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]

                            # Eq. 17
                            x0_t_hat = x0_t - A_funcs.Lambda(A_funcs.A_pinv(
                                A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y_patch.reshape(y_patch.size(0), -1)
                            ).reshape(x0_t.size(0), -1), at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta).reshape(*x0_t.size())
                            # print('x0_t_hat.shape',x0_t_hat.shape)

                            # mask-shift trick 
                            if shift_w == 0 and shift_h == 0:
                                pass
                            elif shift_w == 0 and shift_h != 0:
                                neo_h_l = int(64*shift_h)
                                neo_h_r = neo_h_l+64
                                x0_t_hat[:,:,0:64,:] = x_temp[:,:,neo_h_l:neo_h_r,0:128].to('cuda')
                            else:
                                neo_w_l = int(64*shift_w)
                                neo_w_r = neo_w_l+64
                                neo_h_l = int(64*shift_h)
                                neo_h_r = neo_h_l+128
                                x0_t_hat[:,:,:,0:64] = x_temp[:,:,neo_h_l:neo_h_r,neo_w_l:neo_w_r].to('cuda')
                                if shift_h !=0:
                                    neo_h_r = neo_h_l+64
                                    neo_w_r = neo_w_l+128
                                    x0_t_hat[:,:,0:64,:] = x_temp[:,:,neo_h_l:neo_h_r,neo_w_l:neo_w_r].to('cuda')


                            # Eq. 51
                            xt_next = at_next.sqrt() * x0_t_hat + A_funcs.Lambda_noise(
                                torch.randn_like(x0_t).reshape(x0_t.size(0), -1), 
                                at_next.sqrt()[0, 0, 0, 0], sigma_y, sigma_t, eta, et.reshape(et.size(0), -1)).reshape(*x0_t.size())

                            x0_preds.append(x0_t.to('cpu'))
                            xs.append(xt_next.to('cpu'))

                        else: # time-travel back
                            next_t = (torch.ones(n) * j).to(x.device)
                            at_next = compute_alpha(b, next_t.long())
                            x0_t = x0_preds[-1].to('cuda')
                                
                            xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                            xs.append(xt_next.to('cpu'))

                    finalresult[:,:,int(64*shift_h):int(64*shift_h)+128,int(64*shift_w):int(64*shift_w)+128] = xs[-1]
                    pbar.update(1)


    return [finalresult]

# form RePaint
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
