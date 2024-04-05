import torch
import torch.nn as nn
import torch.nn.functional as func

from einops import rearrange

""" 自定义网络模块
在定义自已的网络的时候，需要继承nn.Module类，并重新实现构造函数__init__和forward这两个方法。但有一些注意技巧：
（1）一般把网络中具有可学习参数的层（如全连接层、卷积层等）放在构造函数__init__()中，当然也可以把不具有参数的层也放在里面；
（2）一般把不具有可学习参数的层(如ReLU、dropout、BatchNorm层)可放在构造函数中，也可不放在构造函数中，如果不放在构造函数__init__里面，则在forward方法里面可以使用nn.functional来代替
（3）forward方法是必须要重写的，它是实现模型的功能，实现各个层之间的连接关系的核心
"""
""" 模块中的forward方法
使用pytorch的时候，模型训练时，不需要使用forward，只要在实例化一个对象中传入对应的参数就可以自动调用forward函数
自动调用forward函数原因分析：
利用Python的语言特性，y = model(x)是调用了对象model的__call__方法，而nn.Module把__call__方法实现为类对象的forward函数，
所以任意继承了nn.Module的类对象都可以这样简写来调用forward函数
调用forward方法的具体流程是：
执行class LeNet(nn.Module); model = LeNet(); y = model(x)
执行时，由于类继承了Module类，而Module这个基类中定义了__call__方法，所以会执行__call__方法，而__call__方法中调用了forward()方法
只要定义类型的时候，实现__call__函数，这个类型就成为可调用的，换句话说，我们可以把这个类型的对象当作函数来使用
"""


class DropPath(nn.Module):
    """
    对batch的dropout，用于窗自注意力模块中，每个残差连接加和之前
    Args:
        drop_prob (float): drop的概率
    """

    def __init__(self, drop_prob: float = 0.):
        super().__init__()  # 继承父类的init方法，防止覆盖父类的init方法
        self.drop_prob = drop_prob  # drop的概率，浮点数

    def forward(self, x):  # 定义前向传播函数
        # 在非训练或者drop的概率为0时，直接返回x
        if self.drop_prob == 0. or not self.training:
            return x
        # 计算drop的结果
        keep_prob = 1 - self.drop_prob  # 保留的比例
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (batch, 1, 1, 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 随机产生0~1再加上keep_prob
        random_tensor.floor_()  # 二值化后的mask，并不严格的按比例drop，有一定的随机性
        x = x.div(keep_prob) * random_tensor  # drop的同时进行放缩，保证总和近乎不变
        return x


class PatchEmbedding(nn.Module):
    """
    利用一次kernel与步长等大的卷积操作，一步实现PatchPartition和LinearEmbedding
    Args:
        patch_size (int): 需要下采样的倍数
        in_c (int): 输入的维度数
        embed_dim (int): embedding后的维度，即论文中的C
        norm_layer (nn.Module): 正则化函数
    """

    def __init__(self, patch_size: int = 4, in_c: int = 3, embed_dim: int = 96, norm_layer: nn.Module = None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(patch_size,) * 2, stride=(patch_size,) * 2)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def padding(self, x: torch.Tensor) -> torch.Tensor:
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        _, _, H, W = x.shape  # [B, C, H, W]
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            x = func.pad(x, (0, self.patch_size - W % self.patch_size,
                             0, self.patch_size - H % self.patch_size,
                             0, 0))  # (W_left, W_right, H_top, H_bottom, C_front, C_back)
        return x

    def forward(self, x):
        x = self.padding(x)  # 按照需要进行padding
        x = self.proj(x)  # 下采样patch_size倍，维度变为embed_dim
        x = rearrange(x, 'B C H W -> B H W C')  # 整理x
        x = self.norm(x)  # 正则化
        return x


class PatchMerging(nn.Module):
    """
    PatchMerging层，每个4*4范围中相同位置的像素组合成一个新的patch，这样会使得长宽减半，维度变4倍，
    模仿卷积网络中降采样升维升2倍的惯例，再经过线性层使维度减半
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)  # 正则化在维度压缩前，故输入是维度的4倍
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 使维度由4倍变为2倍

    @staticmethod  # 声明这里是类的静态函数
    def padding(x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        if H % 2 == 1 or W % 2 == 1:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = func.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        return x

    @staticmethod
    def merging(x: torch.Tensor) -> torch.Tensor:
        # 取对应位置的像素组成新的patch并调整维度
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        return x

    def forward(self, x):
        x = self.padding(x)  # 按需要进行padding
        x = self.merging(x)  # 进行merging
        x = self.norm(x)  # 进行正则化
        x = self.reduction(x)  # 维度减半[B, H/2, W/2, 2*C]
        return x


class PatchExpanding(nn.Module):
    """
    PatchExpanding层，先通过线性层将通道数翻倍，再使用重新排列使长宽翻倍，最终通道数变为原来的一半
    [B, H, W, C] -> [B, H, W, 2*C] -> [B, 2*H, 2*W, C/2]
    """

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=2, P2=2)
        x = self.norm(x)
        return x


class FinalPatchExpanding(nn.Module):
    """
    FinalPatchExpanding层，先通过线性层将通道数翻16倍，再使用重新排列使长宽翻4倍，最终通道数不变
    [B, H, W, C] -> [B, H, W, 16*C] -> [B, 4*H, 4*W, C]
    """

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super(FinalPatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=4, P2=4)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """
    MLP层，即带Dropout的多层感知器(Multilayer perceptron)，也即一个单隐层的全连接神经网络
    Args:
        in_features (int): 输入层特征数
        hidden_features (int): 隐藏层特征数
        out_features (int): 输出层特征数
        act_layer (nn.Module): 激活函数
        drop (float): Dropout比例
    """

    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None,
                 act_layer: nn.Module = nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features  # 当out_features为None时，赋值为in_features，否则赋值为out_features
        hidden_features = hidden_features or in_features  # 当hidden_features为None时，赋值为in_features，否则赋值为hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    """
    加窗多头自注意力模块Window based multi-head self attention (W-MSA) module，带有包含相对位置信息的偏置
    同时也支持移动窗口的加窗多头自注意力Shifted window based multi-head self attention (SW-MSA)
    具体为W-MSA还是SW-MSA，由参数shift指定是否需要移动窗口来决定
    Args:
        dim (int): Number of input channels.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        shift (bool): 是否要计算移动窗口的自注意力
    """

    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0., proj_drop: float = 0., shift: bool = False):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # 如果需要计算移动窗口，则窗口的移动量是窗口大小的一半
        if shift:
            self.shift_size = window_size // 2
        else:
            self.shift_size = 0

        # 定义相对位置偏差的参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))  # [2*Mh-1 * 2*Mw-1, nH]
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # 获取窗口内每个标记成对的相对位置索引（后续相对位置编码使用）
        coords_size = torch.arange(self.window_size)  # 生成参数大小维度的Tensor，值是自然数
        coords = torch.stack(torch.meshgrid([coords_size, coords_size]))  # parameter indexing not in torch 1.8.1
        # coords = torch.stack(torch.meshgrid([coords_size, coords_size], indexing='ij'))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw] = [2, Mh*Mw, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def window_partition(self, x: torch.Tensor) -> torch.Tensor:  # 划分成小的互不重叠的window
        _, H, W, _ = x.shape
        # 划分为不同的块并组合
        x = rearrange(x, 'B (Nh Mh) (Nw Mw) C -> (B Nh Nw) Mh Mw C', Nh=H // self.window_size, Nw=W // self.window_size)
        return x  # [num_windows*B, Mh, Mw, C]

    def create_mask(self, x: torch.Tensor) -> torch.Tensor:  # calculate attention mask for SW-MSA
        _, H, W, _ = x.shape
        # 断言window_size可以整除图像长宽
        assert H % self.window_size == 0 and W % self.window_size == 0, "H or W is not divisible by window_size"

        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, H, W, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self.window_partition(img_mask)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        _, H, W, _ = x.shape  # [B, H, W, C]

        # 使用循环移位(cyclic shift)可以降低SW-MSA的计算复杂度
        if self.shift_size > 0:
            # 循环移位
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # 生成mask
            mask = self.create_mask(x)
        else:
            mask = None

        # 划分成小的互不重叠的window
        x = self.window_partition(x)  # [num_windows*B, Mh, Mw, C]

        # 对x的通道进行调整，方便后续计算
        Bn, Mh, Mw, _ = x.shape  # [num_windows*B, Mh, Mw, C]
        x = rearrange(x, 'Bn Mh Mw C -> Bn (Mh Mw) C')  # [num_windows*B, Mh*Mw, total_embed_dim]

        # 使用线性层计算query, key, value组成的矩阵
        # qkv(x): [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim] ->
        # [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = rearrange(self.qkv(x), 'Bn L (T Nh P) -> T Bn Nh L P', T=3, Nh=self.num_heads)

        # 得到query, key, value
        # 均为 [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # 对维度0进行拆分

        # 计算query和key的相关性
        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # @ 定义为Tensor的乘法

        # 计算相对位置编码
        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]

        # 向相关性矩阵中加入相对位置编码
        attn = attn + relative_position_bias.unsqueeze(0)

        # 如果mask不为None，则向attn中加入mask，mask是包含0和-100的数组，后续进行softmax时，较大的负值对应的权重将变为0
        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.un_squeeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(Bn // nW, nW, self.num_heads, Mh * Mw, Mh * Mw) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, Mh * Mw, Mh * Mw)

        # 将attn做softmax，将相关性矩阵变为权重矩阵，并随机drop其中一些值
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # 根据权重矩阵和value计算最后输出x，并经过线性层和dropout，最后变为[B, H, W, C]
        x = attn @ v  # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        x = rearrange(x, 'Bn Nh (Mh Mw) C -> Bn Mh Mw (Nh C)', Mh=Mh)  # [batch_size*num_windows, Mh, Mw, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, '(B Nh Nw) Mh Mw C -> B (Nh Mh) (Nw Mw) C', Nh=H // Mh, Nw=H // Mw)

        # reverse cyclic shift 反循环移位，计算之后将特征图还原
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return x


class SwinTransformerBlock(nn.Module):
    """
    单个SwinTransformer块，一层SwinTransformer由多个块组成（取决于定义该层的depth）
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift (bool): Shift or not.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift=False, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop, shift=shift)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # Mlp中隐藏层大小是输出层的mlp_ratio倍
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_copy = x  # 拷贝当前值
        x = self.norm1(x)  # 正则化

        x = self.attn(x)  # W-MSA/SW-MSA
        x = self.drop_path(x)  # drop_path
        x = x + x_copy  # 残差链接

        x_copy = x  # 拷贝当前值
        x = self.norm2(x)  # 正则化

        x = self.mlp(x)  # 多层感知机
        x = self.drop_path(x)  # drop_path
        x = x + x_copy  # 残差链接
        return x


class BasicBlock(nn.Module):
    """
    一个封装的基础模块stage，其中包括一层SwinTransformer和前面一层PatchMerging（可选，由参数downsample控制）
    Args:
        index (int): 从0开始的序号，指示是网络中的第几个基础模块，部分参数根据该序号自动选取
        embed_dim (int): embedding后的维度，原论文中的C
        window_size (int): 窗口大小
        depths (tuple[int]): 各个stage中SwinTransformer的深度
        num_heads (tuple[int]): 各个stage中SwinTransformer中多头自注意力的头数
        mlp_ratio (float): 多层感知机中，隐层神经元是输入层的mlp_ratio倍
        qkv_bias (bool): 是否加入可学习的偏置给query, key, value
        drop_rate (float): 用于根据query, key, value计算得到的最终结果x的dropout
        attn_drop_rate (float): 用于根据query, key计算得到的权重矩阵的dropout
        drop_path (float): 在所有的SwinTransformer层中的每个小块中的残差链接中，对batch的dropout，顺序递增到该设定值
        norm_layer (nn.Module): 正则化层
        patch_merging (bool): 是否需要PatchMerging
    """

    def __init__(self, index: int, embed_dim: int = 96, window_size: int = 7, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 norm_layer: nn.Module = nn.LayerNorm, patch_merging: bool = True):
        super(BasicBlock, self).__init__()
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]
        # drop_path_rate从0递增到设定值
        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]
        # 按照论文原代码格式，将该stage中的多个SwinTransformerBlock封装为名叫blocks的ModuleList
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_head,
                window_size=window_size,
                shift=False if (i % 2 == 0) or index == 3 else True,  # 最后一层尺寸不够移窗
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        # 根据需要定义该stage中的patch_merging层，并按照论文原代码的名称，命名成downsample
        if patch_merging:
            self.downsample = PatchMerging(dim=embed_dim * 2 ** index, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicBlockUp(nn.Module):
    """
    一个封装的基础模块stage，其中包括一层SwinTransformer和后面一层PatchExpanding
    （不包含第一层PatchExpanding和最后一层Final_patch_expanding）
    Args:
        index (int): 从0开始的序号，指示是网络中的第几个基础模块，部分参数根据该序号自动选取
        embed_dim (int): embedding后的维度，原论文中的C
        window_size (int): 窗口大小
        depths (tuple[int]): 各个stage中SwinTransformer的深度
        num_heads (tuple[int]): 各个stage中SwinTransformer中多头自注意力的头数
        mlp_ratio (float): 多层感知机中，隐层神经元是输入层的mlp_ratio倍
        qkv_bias (bool): 是否加入可学习的偏置给query, key, value
        drop_rate (float): 用于根据query, key, value计算得到的最终结果x的dropout
        attn_drop_rate (float): 用于根据query, key计算得到的权重矩阵的dropout
        drop_path (float): 在所有的SwinTransformer层中的每个小块中的残差链接中，对batch的dropout，顺序递增到该设定值
        patch_expanding (bool): 是否包含后面的一层PatchExpanding，最后一个stage中不包含
        norm_layer (nn.Module): 正则化层
    """

    def __init__(self, index: int, embed_dim: int = 96, window_size: int = 7, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path: float = 0.1,
                 patch_expanding: bool = True, norm_layer: nn.Module = nn.LayerNorm):
        super(BasicBlockUp, self).__init__()
        index = len(depths) - index - 2
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]
        # drop_path_rate从0递增到设定值
        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]
        # 按照论文原代码格式，将该stage中的多个SwinTransformerBlock封装为名叫blocks的ModuleList
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_head,
                window_size=window_size,
                shift=False if (i % 2 == 0) else True,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        if patch_expanding:
            self.upsample = PatchExpanding(dim=embed_dim * 2 ** index, norm_layer=norm_layer)
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        x = self.upsample(x)
        return x


class SwinUnet(nn.Module):
    """
    Args:
        patch_size (int): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        window_size (int): Window size. Default: 7
        depths (tuple): Depth of each Swin Transformer layer.
        num_heads (tuple): Number of attention heads in different layers.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
    """

    def __init__(self, patch_size: int = 4, in_chans: int = 3, num_classes: int = 1000, embed_dim: int = 96,
                 window_size: int = 7, depths: tuple = (2, 2, 6, 2), num_heads: tuple = (3, 6, 12, 24),
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, norm_layer: nn.Module = nn.LayerNorm, patch_norm: bool = True):
        super().__init__()
        # 定义参数
        self.window_size = window_size
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path = drop_path_rate
        self.norm_layer = norm_layer

        # 定义网络结构
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.layers = self.build_layers()
        self.norm = norm_layer(embed_dim * 2 ** (self.num_layers - 1))
        self.first_patch_expanding = PatchExpanding(dim=embed_dim * 2 ** (len(depths) - 1), norm_layer=norm_layer)
        self.layers_up = self.build_layers_up()
        self.skip_connection_layers = self.skip_connection()
        self.norm_up = norm_layer(embed_dim)
        self.final_patch_expanding = FinalPatchExpanding(dim=embed_dim, norm_layer=norm_layer)
        self.head = nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=(1, 1), bias=False)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):  # 自定义权重的初始化方式
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def build_layers(self):  # 构建网络核心部分的代码层，按照论文原代码命名为layers
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)  # PatchMerging比SwinTransformer少一层
            layers.append(layer)
        return layers

    def build_layers_up(self):  # 构建解码器部分的代码层，命名为layers_up
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):  # 解码器有三层stage
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer)
            layers_up.append(layer)
        return layers_up

    def skip_connection(self):  # 构建跳跃连接的线性层
        skip_connection_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            dim = self.embed_dim * 2 ** (self.num_layers - 2 - i)
            layer = nn.Linear(dim * 2, dim)
            skip_connection_layers.append(layer)
        return skip_connection_layers

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        x_save = []
        for i, layer in enumerate(self.layers):
            x_save.append(x)
            x = layer(x)

        x = self.norm(x)

        x = self.first_patch_expanding(x)

        for i, layer in enumerate(self.layers_up):
            x = torch.cat([x, x_save[len(x_save) - i - 2]], -1)
            x = self.skip_connection_layers[i](x)
            x = layer(x)

        x = self.norm_up(x)
        x = self.final_patch_expanding(x)

        x = rearrange(x, 'B H W C -> B C H W')
        x = self.head(x)
        return x
