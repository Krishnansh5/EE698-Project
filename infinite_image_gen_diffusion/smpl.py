import torch as th
from collections import defaultdict
import tqdm
from typing import Callable
import math
import numpy as np
def sampling(  
    x: th.Tensor,
    rev_ts: th.Tensor,
    noise_fn: Callable,
    x0_pred_fn: Callable,
    round_sigma,
    s_churn: float = 0.0,
    before_step_fn: Callable = None,
    is_tqdm: bool = True,
    return_traj: bool = True,
    device=th.device('cuda'),
    
):
    
    measure_loss = defaultdict(list)
    traj = defaultdict(list)
    if callable(x):
        x = x()
    if traj:
        traj["xt"].append(x.cpu())

    s_t_min = 0.05
    s_t_max = 50.0
    s_noise = 1.003
    rho=7
    S_churn=40
    num_steps = len(rev_ts)
    #print("len rev_ts",len(rev_ts))
    step_indices = th.arange(num_steps, dtype=th.float64, device=device)
    t_steps = (s_t_max ** (1 / rho) + step_indices / (num_steps - 1) * (s_t_min ** (1 / rho) - s_t_max ** (1 / rho))) ** rho
    t_steps = th.cat([round_sigma(t_steps), th.zeros_like(t_steps[:1])])
    #eta = min(s_churn / len(rev_ts), math.sqrt(2.0) - 1)
    
    loop = zip(rev_ts[:-1], rev_ts[1:])
    # if is_tqdm:
    #     loop = tqdm(loop)
    #print("x", x.shape)
    xt_next = x.to(th.float64) * t_steps[0]
    for i, (cur_t, next_t) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'):
        # cur_x = traj["xt"][-1].clone().to("cuda")
        cur_x = xt_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if s_t_min <= cur_t <= s_t_max else 0
        t_hat = round_sigma(cur_t + gamma * cur_t)
        x_hat = cur_x + (t_hat ** 2 - cur_t ** 2).sqrt() * s_noise * th.randn_like(cur_x)
        
        # if cur_t < s_t_max and cur_t > s_t_min:
        #     hat_cur_t = cur_t + eta * cur_t
        #     cur_noise = noise_fn(cur_x, cur_t)
        #     cur_x = cur_x + s_noise * cur_noise * th.sqrt(hat_cur_t ** 2 - cur_t ** 2)
        #     cur_t = hat_cur_t

        if before_step_fn is not None:
            cur_x = before_step_fn(x_hat, t_hat)
        
        x0, loss_info, traj_info = x0_pred_fn(x_hat, t_hat)
        x0 = x0.to(th.float64)
        epsilon_1 = (x_hat - x0) / t_hat

        xt_next = x_hat + (next_t - t_hat) * epsilon_1
        
        if i < num_steps - 1:
            x0, loss_info, traj_info = x0_pred_fn(xt_next, next_t)
            x0 = x0.to(th.float64)
            epsilon_2 = (xt_next - x0) / next_t
            xt_next = x_hat + (next_t - t_hat) * (0.5 * epsilon_1 + 0.5 * epsilon_2)

        # x0, loss_info, traj_info = x0_pred_fn(xt_next, next_t)
        # epsilon_2 = (xt_next - x0) / next_t

        # xt_next = cur_x + (next_t - cur_t) * (epsilon_1 + epsilon_2) / 2

        #running_x = xt_next
        


 
    # x_next = latents.to(torch.float64) * t_steps[0]
    # for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'): # 0, ..., N-1
    #     x_cur = x_next

    #     # Increase noise temporarily.
    #     gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    #     t_hat = net.round_sigma(t_cur + gamma * t_cur)
    #     x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

    #     # Euler step.
    #     #print("x_hat shape", x_hat.shape)
    #     denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
    #     d_cur = (x_hat - denoised) / t_hat
    #     x_next = x_hat + (t_next - t_hat) * d_cur

    #     # Apply 2nd order correction.
    #     if i < num_steps - 1:
    #         denoised = net(x_next, t_next, class_labels).to(torch.float64)
    #         d_prime = (x_next - denoised) / t_next
    #         x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)




        if return_traj:
            for key, value in loss_info.items():
                measure_loss[key].append(value)

            for key, value in traj_info.items():
                traj[key].append(value)
            traj["xt"].append(xt_next.to("cpu").detach())

    if return_traj:
        return traj, measure_loss
    return xt_next