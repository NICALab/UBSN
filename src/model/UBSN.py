import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import random
from .convhole import ConvHole2D
from . import regist_model

@regist_model
class UBSN(nn.Module):
    """
    Blindspot network
    """
    def __init__(self, pd, depth=5, in_ch=1, blind_conv_channels=128, one_by_one_channels=[48, 24, 8, 1], bs_size=1, deep_0=1, deep_1=1, deep_2=0):
        super(UBSN, self).__init__()

        self.pd = pd

        # check arguments
        if type(blind_conv_channels) != int:
            raise Exception("type of blind_conv_channels must be an integer")
        if not all([type(i)==int for i in one_by_one_channels]):
            raise Exception("one_by_one_channels must be an integer array")

        self.depth = depth        
        self.blind_conv_channels = blind_conv_channels
        self.one_by_one_channels = one_by_one_channels
        self.in_ch = in_ch

        self.deep_0 = deep_0
        self.deep_1 = deep_1
        self.deep_2 = deep_2
        
        if type(bs_size) == int:
            bs_size = [bs_size, bs_size]
        self.bs_size = bs_size

        if self.bs_size[0] != self.bs_size[1]:
            raise Exception("bs_size must be a square")
        
        # initialize
        self.relu = nn.ReLU()
        # self.leaky_relu = nn.LeakyReLU()
        self.maxpool_2d =  nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample_2d = nn.Upsample(scale_factor=2)

        # generate bsnet
        self._gen_bsnet()
            

    def _gen_bsnet(self):
        """
        u-net layers -> notblind_layers[], blind_conv3x3_layers[], up_layers[], out_convs[]
        """

        # notblind_layers
        notblind_layers = []
        for d in range(2): #get input as image, two 3x3 conv with relu, 0th level ##0
            c_in = self.in_ch if d == 0 else self.blind_conv_channels
            notblind_layers.append( 
                nn.Conv2d(
                    c_in,
                    self.blind_conv_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    padding_mode="zeros",
                )
            )
            notblind_layers.append(self.relu)
        for d in range(self.depth): #downward, maxpool and 3x3 conv with relu
            notblind_layers.append(self.maxpool_2d)
            notblind_layers.append(
                nn.Conv2d(
                    self.blind_conv_channels,
                    self.blind_conv_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    padding_mode="zeros",
                )
            )
            notblind_layers.append(self.relu)
        self.notblind_layers = nn.ModuleList(notblind_layers)

        # deeper_layers for 0th, 1st level
        deeper_layers = []
        for d in range(self.deep_0+self.deep_1+self.deep_2): #deeper, 3x3 conv with relu
            deeper_layers.append(
                nn.Conv2d(
                    self.blind_conv_channels,
                    self.blind_conv_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                    padding_mode="zeros",
                )
            )
            deeper_layers.append(self.relu)
        self.deeper_layers = nn.ModuleList(deeper_layers)
        
        # blind_conv3x3_layers
        blind_conv3x3_layers = []
        dil = 3
        for d in range(self.depth+1): #blind conv3x3 with relu
            dil = 3+(self.bs_size[0]+(2**d-1))//(2**(d+1))
            if d == 0:
                dil = dil + self.deep_0
            elif d == 1:
                dil = dil + self.deep_1
            elif d == 2:
                dil = dil + self.deep_2

            blind_conv3x3_layers.append(
                ConvHole2D(
                        self.blind_conv_channels,
                        self.blind_conv_channels,
                        kernel_size=3,
                        stride=1,
                        padding=dil, # padding=3,
                        bias=True,
                        padding_mode="zeros",
                        dilation=dil, # dilation=3,
                    )
            )
            blind_conv3x3_layers.append(self.relu)
        self.blind_conv3x3_layers = nn.ModuleList(blind_conv3x3_layers)

        # up_layers
        up_layers = []
        for d in range(self.depth): #upward, upsample and conv1x1 with relu
            up_layers.append(self.upsample_2d)
            c_in = (self.blind_conv_channels)*2 if d == 0 else (self.blind_conv_channels)*3
            up_layers.append(
                nn.Conv2d(
                    c_in,
                    (self.blind_conv_channels)*2,
                    kernel_size=1,
                    padding=0,
                    bias=True,
                )
            )
            up_layers.append(self.relu)
        self.up_layers = nn.ModuleList(up_layers)

        # out_convs
        out_convs = []
        for idx, c in enumerate(self.one_by_one_channels):
            c_in = (
                (self.blind_conv_channels)*2
                if idx == 0
                else self.one_by_one_channels[idx - 1]
            )
            c_out = (
                self.in_ch
                if idx == len(self.one_by_one_channels) - 1
                else c
            )
            out_convs.append(
                nn.Conv2d(
                    c_in,
                    c_out,
                    kernel_size=1,
                    padding=0,
                    bias=True,
                )
            )
            if idx != len(self.one_by_one_channels)-1:
                out_convs.append(self.relu)
        self.out_convs = nn.ModuleList(out_convs)


    def forward_bsnet(self, x):
        
        blind_x = []

        x = self.notblind_layers[0](x) # depth=0
        x = self.notblind_layers[1](x)
        x = self.notblind_layers[2](x)
        x = self.notblind_layers[3](x)
        
        for d in range(self.deep_0):
            if d==0:
                x0 = self.deeper_layers[d*2](x)
            else:
                x0 = self.deeper_layers[d*2](x0)
            x0 = self.deeper_layers[d*2+1](x0)
        x0 = self.blind_conv3x3_layers[0](x0)
        x0 = self.blind_conv3x3_layers[1](x0)
        blind_x.append(x0)

        for d in range(self.depth):
            x = self.notblind_layers[d*3+4](x)
            x = self.notblind_layers[d*3+5](x)
            x = self.notblind_layers[d*3+6](x)
            if d==0:
                for d_ in range(self.deep_1):
                    if d_==0:
                        xn = self.deeper_layers[2*self.deep_0+d_*2](x)
                    else:
                        xn = self.deeper_layers[2*self.deep_0+d_*2](xn)
                    xn = self.deeper_layers[2*self.deep_0+d_*2+1](xn)
                if self.deep_1>0:
                    xn = self.blind_conv3x3_layers[d*2+2](xn)
                else:
                    xn = self.blind_conv3x3_layers[d*2+2](x)
            elif d==1:
                for d_ in range(self.deep_2):
                    if d_==0:
                        xn = self.deeper_layers[2*self.deep_0+2*self.deep_1+d_*2](x)
                    else:
                        xn = self.deeper_layers[2*self.deep_0+2*self.deep_1+d_*2](xn)
                    xn = self.deeper_layers[2*self.deep_0+2*self.deep_1+d_*2+1](xn)
                if self.deep_2>0:
                    xn = self.blind_conv3x3_layers[d*2+2](xn)
                else:
                    xn = self.blind_conv3x3_layers[d*2+2](x)
            else:
                xn = self.blind_conv3x3_layers[d*2+2](x)
            xn = self.blind_conv3x3_layers[d*2+3](xn)
            blind_x.append(xn)

        x = xn
        for d in range(self.depth):
            x = self.up_layers[d*3](x)
            x = torch.cat([x, blind_x[self.depth-1-d]], dim=1)
            x = self.up_layers[d*3+1](x)
            x = self.up_layers[d*3+2](x)

        for o_m in self.out_convs:
            x = o_m(x)

        return x


    def forward(self, x, refine=False): 
        # x = [b, T, d1, d2]
        bsnet_in = x

        avg_n = 2
        bsnet_ins = []
        if avg_n>1:
            for _ in range(avg_n):
                _bsnet_in = x.clone()
                bsnet_ins.append(_bsnet_in)

        randomizedpd = False
        testing = False
        
        if self.training:
            pd = random.randint(3, 5) # 4 # self.pd[0]
            randomizedpd = True
        elif refine:
            pd = self.pd[2]
        else:
            testing = True # False
            pd = self.pd[1]
        
        if randomizedpd:
            f = pd
            b = x.shape[0]
            rotate_list = [random.randint(0, 3) for _ in range(b*f*f)] # [0 for _ in range(b*f*f)]
            order_list = list(range(f*f))
            # random.shuffle(order_list)
            subsample_orders = []
            for i in range(x.shape[2]//f):
                for j in range(x.shape[3]//f):
                    makelist = list(range(f*f))
                    random.shuffle(makelist)
                    subsample_orders.append(makelist)
        rotate_lists = []
        order_lists = []
        subsample_orders_list = []
        if avg_n>1:
            b = x.shape[0]
            f = pd
            for _ in range(avg_n):
                _rotate_list = [random.randint(0, 3) for _ in range(b*f*f)]
                _order_list = list(range(f*f))
                # random.shuffle(_order_list)
                _subsample_orders = []
                for i in range(x.shape[2]//f):
                    for j in range(x.shape[3]//f):
                        makelist = list(range(f*f))
                        random.shuffle(makelist)
                        _subsample_orders.append(makelist)
                rotate_lists.append(_rotate_list)
                order_lists.append(_order_list)
                subsample_orders_list.append(_subsample_orders)
        if testing:
            f = pd
            b = x.shape[0]
            rotate_list = [random.randint(0, 3) for _ in range(b*f*f)] # [0 for _ in range(b*f*f)]
            order_list = list(range(f*f))
            # random.shuffle(order_list)

        b, c, h, w = bsnet_in.shape
        if pd>1:
            p = 0
            if randomizedpd:
                bsnet_in = Randomized_PD_train_down_sampling(bsnet_in,pd, rotate_list=rotate_list, order_list=order_list, subsample_orders=subsample_orders)                
            elif testing:
                if avg_n>1:
                    for k in range(avg_n):
                        bsnet_ins[k] = Randomized_PD_test_down_sampling(bsnet_ins[k],pd, rotate_list=rotate_lists[k], order_list=order_lists[k])
                else:
                    bsnet_in = Randomized_PD_test_down_sampling(bsnet_in,pd, rotate_list=rotate_list, order_list=order_list)
            else:
                bsnet_in = pixel_shuffle_down_sampling(bsnet_in,pd)   
        else:
            p = 0
            bsnet_in = F.pad(bsnet_in, (p,p,p,p), 'reflect')

        # denoise
        x = self.forward_bsnet(bsnet_in)
        xs = []
        if avg_n>1:
            for k in range(avg_n):
                xs.append(self.forward_bsnet(bsnet_ins[k]))

        if pd>1:
            if randomizedpd:
                x = Randomized_PD_train_up_sampling(x,pd, rotate_list=rotate_list, order_list=order_list, subsample_orders=subsample_orders)
            elif testing:
                if avg_n>1:
                    for k in range(avg_n):
                        xs[k] = Randomized_PD_test_up_sampling(xs[k],pd, rotate_list=rotate_lists[k], order_list=order_lists[k])
                    x = torch.mean(torch.stack(xs), dim=0)
                else:
                    x = Randomized_PD_test_up_sampling(x,pd, rotate_list=rotate_list, order_list=order_list)
            else:
                x = pixel_shuffle_up_sampling(x,pd)
    
        if p == 0:
            x = x
        else:
            x = x[:,:,p:-p,p:-p]

        return x





def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    # batched image tensor
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)   
    # batched image tensor
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)


def Randomized_PD_test_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0., rotate_list=[], order_list=[]):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    b,c,w,h = x.shape
    unshuffled = F.pixel_unshuffle(x, f)

    if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
    spread = unshuffled.view(b,c,f*f,w//f+2*pad,h//f+2*pad)

    for i in range(b):
        for j in range(f*f):
            slice = spread[i, :, j, :, :]
            rotation = rotate_list[i*f*f+j]
            rotated_slice = torch.rot90(slice, rotation, [1, 2])
            spread[i, :, j, :, :] = rotated_slice
    
    changeorder = spread[:, :, order_list, :, :]
    changeorder_spread = changeorder.view(b, c, f, f, w//f+2*pad, h//f+2*pad).permute(0,1,2,4,3,5)
    return changeorder_spread.reshape(b,c,w+2*f*pad, h+2*f*pad)

def Randomized_PD_test_up_sampling(x:torch.Tensor, f:int, pad:int=0, rotate_list=[], order_list=[]):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    b,c,w,h = x.shape
    spread = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5)
    spread_restore = spread.reshape(b,c,f*f, w//f, h//f)
    inverse_order = inversePermutation(order_list)
    # print("inverse_order", inverse_order)
    spread_restore = spread_restore[:, :, inverse_order, :, :]
    for i in range(b):
        for j in range(f*f):
            slice = spread_restore[i, :, j, :, :]
            rotation = rotate_list[i*f*f+j]
            rotated_slice = torch.rot90(slice, -rotation, [1, 2])
            spread_restore[i, :, j, :, :] = rotated_slice
    
    before_shuffle = spread_restore.reshape(b,c*f*f,w//f,h//f)
    if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
    return F.pixel_shuffle(before_shuffle, f)

def inversePermutation(arr):
    size = len(arr)
    arr2 = [0] * size
    for i in range(size):
        arr2[arr[i]] = i
    return arr2


def Randomized_PD_train_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0., rotate_list=[], order_list=[], subsample_orders=[]):

    b,c,w,h = x.shape

    x_s = x.view(b, c, w//f, f, h//f, f).permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, w//f, h//f, f*f)
    subsample_orders = torch.tensor(subsample_orders, device=x.device, dtype=torch.long).view(w//f, h//f, -1)
    x_s = torch.gather(x_s, 4, subsample_orders.expand(b, c, -1, -1, -1))
    x = x_s.view(b, c, w//f, h//f, f, f).permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, w, h)

    unshuffled = F.pixel_unshuffle(x, f)

    spread = unshuffled.view(b,c,f*f,w//f+2*pad,h//f+2*pad)

    for i in range(b):
        for j in range(f*f):
            slice = spread[i, :, j, :, :]
            rotation = rotate_list[i*f*f+j]
            rotated_slice = torch.rot90(slice, rotation, [1, 2])
            spread[i, :, j, :, :] = rotated_slice

    changeorder = spread[:, :, order_list, :, :]
    changeorder_spread = changeorder.view(b, c, f, f, w//f+2*pad, h//f+2*pad).permute(0,1,2,4,3,5)
    return changeorder_spread.reshape(b,c,w+2*f*pad, h+2*f*pad)

def Randomized_PD_train_up_sampling(x:torch.Tensor, f:int, pad:int=0, rotate_list=[], order_list=[], subsample_orders=[]):

    b,c,w,h = x.shape

    spread = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5)
    spread_restore = spread.reshape(b,c,f*f, w//f, h//f)
    inverse_order = inversePermutation(order_list)
    # print("inverse_order", inverse_order)
    spread_restore = spread_restore[:, :, inverse_order, :, :]
    for i in range(b):
        for j in range(f*f):
            slice = spread_restore[i, :, j, :, :]
            rotation = rotate_list[i*f*f+j]
            rotated_slice = torch.rot90(slice, -rotation, [1, 2])
            spread_restore[i, :, j, :, :] = rotated_slice
    
    before_shuffle = spread_restore.reshape(b,c*f*f,w//f,h//f)
    # if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]

    x_s = F.pixel_shuffle(before_shuffle, f)
    x_s = x_s.view(b, c, w//f, f, h//f, f).permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, w//f, h//f, f*f)
    subsample_orders_tensor = torch.tensor([inversePermutation(order) for order in subsample_orders], device=x.device, dtype=torch.long).view(w//f, h//f, -1)
    x_s = torch.gather(x_s, 4, subsample_orders_tensor.expand(b, c, -1, -1, -1))
    x = x_s.view(b, c, w//f, h//f, f, f).permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, w, h)
    return x
