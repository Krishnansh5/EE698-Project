import torch as th
from einops import rearrange

__all__ = [
    "split_wimg",
]

def split_wimg(wimg, n_img, rtn_overlap=True):
    if wimg.ndim == 3:
        wimg = wimg[None]
    _, _, h, w = wimg.shape
    overlap_size = (n_img * h - w) // (n_img - 1)
    assert n_img * h - overlap_size * (n_img - 1) == w
    
    img = th.nn.functional.unfold(wimg, kernel_size=(h,h), stride=h - overlap_size) 
    img = rearrange(
        img,
        "b (c h w) n -> (b n) c h w", h=h, w=h
    )
    
    if rtn_overlap:
        return img , overlap_size
    return img

def avg_merge_wimg(imgs, overlap_size, n=None, is_avg=True):
    b, _, h, w = imgs.shape
    if n == None:
        n = b
    unfold_img = rearrange(
        imgs,
        "(b n) c h w -> b (c h w) n", n = n
    )
    img = th.nn.functional.fold(
        unfold_img,
        (h, n * w - (n-1) * overlap_size),
        kernel_size = (h, w),
        stride = h - overlap_size
    ) 
    if is_avg:
        counter = th.nn.functional.fold(
            th.ones_like(unfold_img), 
            (h, n * w - (n-1) * overlap_size),
            kernel_size = (h, w),
            stride = h - overlap_size
        )
        return img / counter
    return img

