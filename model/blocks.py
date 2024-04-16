
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


from functools import reduce
from typing import Union, Tuple, MutableSequence, List, Optional

from torch import Tensor
from torch.nn import Conv2d, BatchNorm2d
from torch.nn import Module, Conv1d, BatchNorm1d, LeakyReLU

# from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
import torch
import math
import warnings


class FinalModel(nn.Module):
    def __init__(self, num_classes=26):
        super(FinalModel, self).__init__()
        self.is_train = True
        self.num_classes = num_classes
        self.max_len = 1152

        self.blocks = nn.Sequential(
            Block(1, 24, 2, 4),
            Block(24, 48, 2, 5),
            Block(48, 96, 2, 6),
            Block(96, 96, 1, 7),
            nn.Conv2d(96, 128, kernel_size=1))

        if self.is_train == True:
            self.mask = False
        else:
            self.mask = False
        self.embd_dim = 128
        self.transformer = ConvTransformerBackbone(n_in=128,
                                                   n_embd=self.embd_dim,
                                                   n_head=4,
                                                   n_embd_ks=5,
                                                   max_len=self.max_len,
                                                   mha_win_size=[
                                                       19]*6 + [-1]*3,
                                                   arch=(2, 2, 8),
                                                   use_abs_pe=False)
        self.linearN = nn.Sequential(nn.Linear(60, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 1))
        self.linear1 = nn.Linear(self.embd_dim, 128)
        self.linear2 = nn.Linear(128, self.num_classes)

    def forward(self, x, l):
        x = self.blocks(x.transpose(1, 2))
        x = F.upsample(x, size=(12, 1152), mode='bilinear')

        x = self.transformer(x)
        x = F.leaky_relu(self.linearN(x.flatten(-2)).squeeze(-1))

        x = F.leaky_relu(self.linear1(x))
        x = self.linear2(x)

        return x


def apply_layer(layer_input: Tensor,
                layer: Module) \
        -> Tensor:
    """Small aux function to speed up reduce operation.
    :param layer_input: Input to the layer.
    :type layer_input: torch.Tensor
    :param layer: Layer.
    :type layer: torch.nn.Module
    :return: Output of the layer.
    :rtype: torch.Tensor
    """
    return layer(layer_input)


class DepthWiseSeparableConvBlock(Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int], MutableSequence[int]],
                 stride: Optional[int] = 1,
                 padding: Optional[int] = 0,
                 dilation: Optional[int] = 1,
                 bias: Optional[bool] = True,
                 padding_mode: Optional[str] = 'zeros',
                 inner_kernel_size: Optional[Union[int,
                                                   Tuple[int, int], MutableSequence[int]]] = 1,
                 inner_stride: Optional[int] = 1,
                 inner_padding: Optional[int] = 0) \
            -> None:

        super(DepthWiseSeparableConvBlock, self).__init__()

        self.depth_wise_conv: Module = Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, padding),
            dilation=(1, dilation), groups=in_channels, bias=bias,
            padding_mode=padding_mode)

        self.non_linearity: Module = LeakyReLU()

        self.batch_norm: Module = BatchNorm2d(out_channels)

        self.point_wise: Module = Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(1, inner_kernel_size), stride=(1, inner_stride),
            padding=(0, inner_padding), dilation=1,
            groups=1, bias=bias, padding_mode=padding_mode)

        self.layers: List[Module] = [
            self.depth_wise_conv,
            self.non_linearity,
            self.batch_norm,
            self.point_wise]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the module.
        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        return reduce(apply_layer, self.layers, x)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, depth, grow=False):
        super(Block, self).__init__()
        assert out_channels % in_channels == 0

        self.p0 = DepthWiseSeparableConvBlock(in_channels, out_channels, kernel_size=9,
                                              stride=1)

        self.p01 = DepthWiseSeparableConvBlock(out_channels, out_channels, kernel_size=9,
                                               stride=stride)
        self.p1 = nn.ModuleList()

        for _ in range(depth):

            self.p1.append(DepthWiseSeparableConvBlock(out_channels, out_channels, kernel_size=9,
                                                       stride=1, padding=4))
        self.p1 = nn.Sequential(*self.p1)

    def forward(self, x):
        x = self.p0(x)
        x = self.p01(x)
        res = x
        x = self.p1(x)
        x = res + x

        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """

    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        leads=2,
        arch=(2, 2, 8),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size=[-1]*9,  # size of local window for mha
        scale_factor=2,      # dowsampling rate for the branch
        with_ln=True,       # if to attach layernorm after conv
        attn_pdrop=0.0,      # dropout rate for the attention map
        proj_pdrop=0.0,      # dropout rate for the projection / MLP
        path_pdrop=0.1,      # droput rate for drop path
        use_abs_pe=True,    # use absolute position embedding
        use_rel_pe=False,    # use relative position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        # feature projection
        self.n_in = n_in

        self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                nn.Conv2d(
                    n_in, n_embd, (1, n_embd_ks),
                    stride=(1, 1), padding=(0, n_embd_ks//2), bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(nn.BatchNorm2d(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(
                self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, L, T = x.size()

        # feature projection

        for idx in range(len(self.embd)):
            x = self.embd[idx](x)
            x = self.relu(self.embd_norm[idx](x))
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T]

        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T]
        # stem transformer

        for idx in range(len(self.stem)):
            x = self.stem[idx](x)
        # prep for outputs
        out_feats = (x, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x = self.branch[idx](x)

            out_feats += (x, )

        return out_feats[-1]


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0

        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)
        # masking the output, stop grad to mask
        out_conv = out_conv
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
        self,
        num_channels,
        eps=1e-5,
        affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


# helper functions for Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i)
                              for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


class MaskedMHCA(nn.Module):

    def __init__(
        self,
        n_embd,          # dimension of the output features
        n_head,          # number of heads in multi-head self-attention
        n_qx_stride=1,   # dowsampling stride for query and input
        n_kv_stride=1,   # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = nn.Conv2d(
            self.n_embd, self.n_embd, (1, kernel_size),
            stride=(1, stride), padding=(0, padding), groups=self.n_embd, bias=False
        )
        self.query_norm = nn.BatchNorm2d(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = nn.Conv2d(
            self.n_embd, self.n_embd, (1, kernel_size),
            stride=(1, stride), padding=(0, padding), groups=self.n_embd, bias=False
        )
        self.key_norm = nn.BatchNorm2d(self.n_embd)
        self.value_conv = nn.Conv2d(
            self.n_embd, self.n_embd, (1, kernel_size),
            stride=(1, stride), padding=(0, padding), groups=self.n_embd, bias=False
        )
        self.value_norm = nn.BatchNorm2d(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv2d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv2d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv2d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv2d(self.n_embd, self.n_embd, 1)

    def forward(self, x):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, L, T = x.size()

        # query conv -> (B, nh * hs, T')
        q = self.query_conv(x)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k = self.key_conv(x)
        k = self.key_norm(k)
        v = self.value_conv(x)
        v = self.value_norm(v)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels,
                   L, -1).transpose(2, 4).flatten(-2)
        q = q.view(B, self.n_head, self.n_channels,
                   L, -1).transpose(2, 4).flatten(-2)
        v = v.view(B, self.n_head, self.n_channels,
                   L, -1).transpose(2, 4).flatten(-2)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)

        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ v
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, L, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out))
        return out


class LocalMaskedMHCA(nn.Module):
    """
    Local Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    The implementation is fairly tricky, code reference from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the output features
        n_head,          # number of heads in multi-head self-attention
        window_size,     # size of the local attention window
        n_qx_stride=1,   # dowsampling stride for query and input
        n_kv_stride=1,   # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
        use_rel_pe=False  # use relative position encoding
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap = window_size // 2
        # must use an odd window size
        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = nn.Conv2d(
            self.n_embd, self.n_embd, (1, kernel_size),
            stride=(1, stride), padding=(0, padding), groups=self.n_embd, bias=False
        )
        self.query_norm = nn.BatchNorm2d(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = nn.Conv2d(
            self.n_embd, self.n_embd, (1, kernel_size),
            stride=(1, stride), padding=(0, padding), groups=self.n_embd, bias=False
        )
        self.key_norm = nn.BatchNorm2d(self.n_embd)
        self.value_conv = nn.Conv2d(
            self.n_embd, self.n_embd, (1, kernel_size),
            stride=(1, stride), padding=(0, padding), groups=self.n_embd, bias=False
        )
        self.value_norm = nn.BatchNorm2d(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv2d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv2d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv2d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv2d(self.n_embd, self.n_embd, 1)

        # relative position encoding
        if self.use_rel_pe:
            self.rel_pe = nn.Parameter(
                torch.zeros(1, 1, self.n_head, self.window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / self.n_embd)**0.5)

    @staticmethod
    def _chunk(x, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # x: B x nh, T, hs
        # non-overlapping chunks of size = 2w -> B x nh, T//2w, 2w, hs
        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2

        # B x nh, #chunks = T//w - 1, 2w, hs
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        """pads rows and then flips rows and columns"""
        # padding value is not important because it will be overwritten
        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(
            affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:,
                                       :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        # `== 1` converts to bool or uint8
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -
                                    affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        # `== 1` converts to bool or uint8
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example::
              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonalize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1).
        x = nn.functional.pad(
            x, (0, window_overlap + 1)
        )
        # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        x = x.view(total_num_heads, num_chunks, -1)
        # total_num_heads x num_chunks x window_overlap*window_overlap
        x = x[:, :, :-window_overlap]
        x = x.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(
        self, query, key, num_heads, window_overlap
    ):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This implementation splits the input into overlapping chunks of size 2w with an overlap of size w (window_overlap)
        """
        # query / key: B*nh, T, hs
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # B * num_heads, head_dim, #chunks=(T//w - 1), 2w
        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (chunk_query, chunk_key))

        # convert diagonals into columns
        # B * num_heads, #chunks, 2w, 2w+1
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1,
             window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1): -1, window_overlap + 1:
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap:
        ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs, value, num_heads, window_overlap
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(
            value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads,
                              chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(
            size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum(
            "bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, L, T = x.size()

        # x = x.permute(0, 2, 1, 3)
       # print('asf.',x.shape)

        # step 1: depth convolutions
        # query conv -> (B, nh * hs, T')
        q = self.query_conv(x)
        q = self.query_norm(q)

        # key, value conv -> (B, nh * hs, T'')
        k = self.key_conv(x)
        k = self.key_norm(k)
        v = self.value_conv(x)
        v = self.value_norm(v)

        # step 2: query, key, value transforms & reshape
        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # x = x.permute(0, 2, 1, 3)
        # (B, nh * hs, T) -> (B, nh, T, hs)
        q = q.view(B, self.n_head, self.n_channels, L,
                   -1).transpose(2, 4).contiguous()
        k = k.view(B, self.n_head, self.n_channels, L,
                   -1).transpose(2, 4).contiguous()
        v = v.view(B, self.n_head, self.n_channels, L,
                   -1).transpose(2, 4).contiguous()
        # view as (B * nh, T, hs)
        q = q.view(B * self.n_head, -1, self.n_channels*L).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels*L).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels*L).contiguous()

        # step 3: compute local self-attention with rel pe and masking
        q *= self.scale
        # chunked query key attention -> B, T, nh, 2w+1 = window_size
        att = self._sliding_chunks_query_key_matmul(
            q, k, self.n_head, self.window_overlap)

        # rel pe
        if self.use_rel_pe:
            att += self.rel_pe
        # kv_mask -> B, T'', 1

        # ignore input masking for now
        att = nn.functional.softmax(att, dim=-1)
        # softmax sometimes inserts NaN if all positions are masked, replace
        att = self.attn_drop(att)

        # step 4: compute attention value product + output projection
        # chunked attn value product -> B, nh, T, hs
        out = self._sliding_chunks_matmul_attn_probs_value(
            att, v, self.n_head, self.window_overlap)
        # transpose to B, nh, hs, T -> B, nh*hs, T
        out = out.transpose(2, 3).contiguous().view(B, C, L, -1)
        # output projection + skip connection
        out = self.proj_drop(self.proj(out))
        return out


class TransformerBlock(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,                # dimension of the input features
        n_head,                # number of attention heads
        n_ds_strides=(1, 1),   # downsampling strides for q & x, k & v
        n_out=None,            # output dimension, if None, set to input dim
        n_hidden=None,         # dimension of the hidden layer in MLP
        act_layer=nn.GELU,     # nonlinear activation used in MLP, default GELU
        attn_pdrop=0.0,        # dropout rate for the attention map
        proj_pdrop=0.0,        # dropout rate for the projection / MLP
        path_pdrop=0.0,        # drop path rate
        mha_win_size=-1,       # > 0 to use window mha
        use_rel_pe=False       # if to add rel position encoding to attention
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        # layer norm for order (B C T)
        self.ln1 = nn.BatchNorm2d(n_embd)
        self.ln2 = nn.BatchNorm2d(n_embd)

        # specify the attention module
        if mha_win_size > 1:
            self.attn = LocalMaskedMHCA(
                n_embd,
                n_head,
                window_size=mha_win_size,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe  # only valid for local attention
            )
        else:
            self.attn = MaskedMHCA(
                n_embd,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop
            )

        # input
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
            self.pool_skip = nn.MaxPool2d(
                (1, kernel_size), stride=(1, stride), padding=(0, padding))
        else:
            self.pool_skip = nn.Identity()

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp = nn.Sequential(
            nn.Conv2d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv2d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        self.drop_path_attn = nn.Identity()
        self.drop_path_mlp = nn.Identity()

    def forward(self, x, pos_embd=None):
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf
        out = self.attn(self.ln1(x))
        out = self.pool_skip(x) + self.drop_path_attn(out)
        # FFN
        out = out + \
            self.drop_path_mlp(self.mlp(self.ln2(out)))
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd
        return out


# drop path: from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)
