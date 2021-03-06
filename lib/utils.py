import torch
import torch.nn as nn
import torchvision
import cv2
import os


def set_norm_layer(norm_type, norm_dim):
    if norm_type == 'bn':
        norm = nn.BatchNorm2d(norm_dim)
    elif norm_type == 'in':
        norm = nn.InstanceNorm2d(norm_dim)
    elif norm_type == 'none':
        norm = None
    else:
        assert 0, "Unsupported normalization: {}".format(norm)
    return norm

def set_activate_layer(types):
    # initialize activation
    if types == 'relu':
        activation = nn.ReLU()
    elif types == 'lrelu':
        activation = nn.LeakyReLU(0.2)
    elif types == 'tanh':
        activation = nn.Tanh()
    elif types == 'sig':
        activation = nn.Sigmoid()
    elif types == 'none':
        activation = None
    else:
        assert 0, f"Unsupported activation: {types}"
    return activation


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
        
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


def update_net(optimizer, loss):
    optimizer.zero_grad()  
    loss.backward()   
    optimizer.step()  


def setup_ddp(gpu, ngpus_per_node):
    torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=ngpus_per_node,
            rank=gpu)


def save_image(args, global_step, dir, images):
    dir_path = f'{args.save_root}/{args.run_id}/{dir}'
    os.makedirs(dir_path, exist_ok=True)
    
    sample_image = make_grid_image(images).transpose([1,2,0]) * 255
    cv2.imwrite(dir_path + f'/e{global_step}.jpg', sample_image[:,:,::-1])


def make_grid_image(images_list):
    grid_rows = []

    for images in images_list:
        images = images[:8] # Drop images if there are more than 8 images in the list
        grid_row = torchvision.utils.make_grid(images.detach().cpu(), nrow=images.shape[0]) * 0.5 + 0.5
        grid_rows.append(grid_row)

    grid = torch.cat(grid_rows, dim=1).numpy()
    return grid
