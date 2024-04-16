import argparse
from PIL import Image
import os
import numpy as np

BBOX_CFG = {
    'shape': (64,64),
    'margin': (10,10),
    'random_size': False
}

FF_CFG = {
    'mv': 60,
    'ma': 3.1415926,
    'ml': 100,
    'mbw': 20
}

OP_CFG = {
    'height': 32, 
    'width': 32,
    'random': False
}

STITCH_CFG = {
    'width': 128,
}

def random_bbox(config, shape):
    """Generate a random tlhw with configuration.
    Args:
        config: Config should have configuration including DATA_NEW_SHAPE,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    Returns:
        tuple: (top, left, height, width)
    """ 
    img_height = shape[0]
    img_width = shape[1]
    height, width = config["shape"]
    ver_margin, hor_margin = config["margin"]
    maxt = img_height - ver_margin - height
    maxl = img_width - hor_margin - width
    t = np.random.randint(low=ver_margin, high=maxt)
    l = np.random.randint(low=hor_margin, high=maxl)
    h = height
    w = width
    return (t, l, h, w)


def random_ff_mask(config, shape):
    """Generate a random free form mask with configuration.
    Args:
        config: Config should have configuration including DATA_NEW_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    Returns:
        tuple: (top, left, height, width)
    """

    h, w = shape
    mask = np.zeros((h, w))
    num_v = 12 + np.random.randint(
        config["mv"]
    )  # tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(config["ma"])
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(config["ml"])
            brush_w = 10 + np.random.randint(config["mbw"])
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y

    return mask.reshape((1,) + mask.shape).astype(np.float32)


def bbox2mask(bboxs, shape, config):
    """Generate mask tensor from bbox.
    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including DATA_NEW_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
    Returns:
        tf.Tensor: output with shape [1, H, W, 1]
    """
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    # print(mask.shape)
    for bbox in bboxs:
        if config["random_size"]:
            h = int(0.1 * bbox[2]) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(0.1 * bbox[3]) + np.random.randint(int(bbox[3] * 0.2) + 1)
        else:
            h = 0
            w = 0
        mask[
            bbox[0] + h : bbox[0] + bbox[2] - h, bbox[1] + w : bbox[1] + bbox[3] - w
        ] = 1.0
    # print("after", mask.shape)
    return mask.reshape((1,) + mask.shape).astype(np.float32)

def out_painting_mask(config,shape):
    h, w = shape
    mask = np.ones((h, w), np.float32)
    if config['random']:
        x = np.random.randint(0, config['width'])
        y = np.random.randint(0, config['height'])
    else :
        x = config['width']
        y = config['height']
    mask[y:h-y, x:w-x] = 0.0
    return mask.reshape((1,) + mask.shape).astype(np.float32)

def stitching_mask(config,shape):
    h, w = shape
    mask = np.zeros((h, w), np.float32)
    w_center = w//2
    m_w = config['width']
    # m_w = w // 3
    mask[:,w_center - m_w//2:w_center + m_w//2] = 1.0
    return mask.reshape((1,) + mask.shape).astype(np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_dir', type=str, default='./stitch_masks/')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--mask_type', type=str, default='sticth')
    parser.add_argument('--num_mask', type=int, default=1)
    args = parser.parse_args()

    # clear all files
    for file in os.listdir(args.out_dir):
        file_path = os.path.join(args.out_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    for i in range(args.num_mask):
        if args.mask_type == 'random_bbox':
            bbox = random_bbox(BBOX_CFG, (args.img_size, args.img_size))
            mask_tensor = bbox2mask([bbox], (args.img_size, args.img_size),BBOX_CFG)
            mask_img = Image.fromarray((mask_tensor[0] * 255).astype('uint8'))
            mask_img.save(args.out_dir + f'/mask_{i}.png')
        elif args.mask_type == 'random_free_form':
            mask_tensor = random_ff_mask(FF_CFG, (args.img_size, args.img_size))
            mask_img = Image.fromarray((mask_tensor[0] * 255).astype('uint8'))
            mask_img.save(args.out_dir + f'/mask_{i}.png')
        elif args.mask_type == 'outpainting':
            mask_tensor = out_painting_mask(OP_CFG, (args.img_size, args.img_size))
            mask_img = Image.fromarray((mask_tensor[0] * 255).astype('uint8'))
            mask_img.save(args.out_dir + f'/mask_{i}.png')
        elif args.mask_type == 'sticth':
            mask_tensor = stitching_mask(STITCH_CFG, (args.img_size, args.img_size))
            mask_img = Image.fromarray((mask_tensor[0] * 255).astype('uint8'))
            mask_img.save(args.out_dir + f'/mask_{i}.png')
