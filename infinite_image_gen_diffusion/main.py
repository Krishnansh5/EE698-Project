import smpl
from einops import rearrange
from w_img import split_wimg, avg_merge_wimg
import numpy as np
import torch 
import torch as th
import pickle
import PIL.Image

class CondIndLong():
    def __init__(self, shape, eps_scalar_t_fn, batch_size, num_img, x_, overlap_size=32, sigma_max=80.0, sigma_min=1e-3):
        c, h, w = shape
        self.overlap_size = overlap_size
        self.num_img = num_img
        self.btch = batch_size
        final_img_w = w * num_img - self.overlap_size * (num_img - 1)
        self.shape = (c, h, final_img_w)
        self.eps_scalar_t_fn = eps_scalar_t_fn
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.x_ = x_
    def loss(self, x):
        x1, x2 = x[:-1], x[1:]
        return th.sum(
            (th.abs(x1[:, :, :, -self.overlap_size :] - x2[:, :, :, : self.overlap_size])) ** 2,
            dim=(1, 2, 3),
        )
    
    def generate_xT(self, n):
        k = self.sigma_max * th.randn((n , *self.shape)).cuda()
        return k
    
    def x0_fn(self, xt, scalar_t, enable_grad=False):
        cur_eps = self.diff(long_x = xt, scalar_t = scalar_t).to(th.float64)
        x0 = xt - scalar_t * cur_eps
        x0 = th.clip(x0, -1,1) 
        return cur_eps, {}, {"x0": x0.cpu()}

    def noise(self, xt, scalar_t):
        del scalar_t
        return th.randn_like(xt)

    def rev_ts(self, n_step, ts_order):
        _rev_ts = th.pow(
            th.linspace(
                np.power(self.sigma_max, 1.0 / ts_order),
                np.power(self.sigma_min, 1.0 / ts_order),
                n_step + 1
            ),
            ts_order
        )
        return _rev_ts.cuda()
    def round(self):
        return self.eps_scalar_t_fn.round_sigma
    def diff(self, long_x, scalar_t, enable_grad=False):
        xs = split_wimg(long_x, self.num_img, rtn_overlap=False)
        c_ = self.x_
        for cc in range(self.num_img-1):
            c_ = torch.cat((c_, self.x_), dim=0)
        full_eps = self.eps_scalar_t_fn(xs, scalar_t, c_) # #((b,n), c, h, w) , self.x_
        full_eps = rearrange(
                full_eps,
                "(b n) c h w -> n b c h w", n = self.num_img
            )
        half_eps = self.eps_scalar_t_fn(xs[:,:,:,-self.overlap_size:], scalar_t, c_).to(th.float64) #((b,n), c, h, w//2) , self.x_
        half_eps = rearrange(
                half_eps,
                "(b n) c h w -> n b c h w", n = self.num_img
            )

        half_eps[-1]=0

        full_eps[:,:,:,:,-self.overlap_size:] = full_eps[:,:,:,:,-self.overlap_size:] - half_eps
        whole_eps = rearrange(
                full_eps,
                "n b c h w -> (b n) c h w"
            )
        return avg_merge_wimg(whole_eps, self.overlap_size, n=self.num_img, is_avg=False)
    
def test( model_path, dest_path, s_churn=10.0, device=torch.device('cuda')):
    n_step = 40
    overlap_size = 32
    batch_size = 3
    ts_order = 5
    num_imgs = 15
    img_resolution = 64
    img_channels = 3
    img_shape = (img_channels, img_resolution, img_resolution)
    print(f'Loading model from "{model_path}"...')
    with open(model_path, 'rb') as f:
       diffusion = pickle.load(f)['ema'].to(device)

    class_labels = None
    print('ll', diffusion.label_dim)
    if diffusion.label_dim:
        class_labels = torch.eye(diffusion.label_dim, device=device)[torch.randint(diffusion.label_dim, size=[batch_size], device=device)]

    worker = CondIndLong(img_shape, diffusion, batch_size, num_imgs, overlap_size=overlap_size, x_ = class_labels)
    sample = smpl.sampling(
        x = worker.generate_xT(batch_size),
        noise_fn = worker.noise,
        rev_ts = worker.rev_ts(n_step, ts_order),
        x0_pred_fn = worker.x0_fn,
        s_churn = s_churn,
        return_traj = False,
        device = device,
        round_sigma = worker.round()    
    )
    print("sample shape", sample.shape)
    print(f'Saving image grid to "{dest_path}"...')
    image = (sample * 127.5 + 128).clip(0, 255).to(torch.uint8)
    print(image.shape)

    image = image.reshape(batch_size, 1, *image.shape[1:]).permute(1, 0, 3, 4, 2)
    print(image.shape)

    image = image.reshape((batch_size )* img_resolution, (num_imgs+1)*(overlap_size), img_channels)
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, 'RGB').save(dest_path)
    print('Done.')
test(model_path='edm-imagenet-64x64-cond-adm.pkl', dest_path='Generated_image.png')
