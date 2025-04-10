import torch
import torch.nn.functional as F
import math


class KANLinearLoRA(torch.nn.Module):
    """
    LoRA 版本的 KANLinear：
    - base_weight + LoRA(base_weight)
    - spline_weight + LoRA(spline_weight) (可选)
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        # ===== LoRA相关参数 =====
        lora_rank_base=1,             # 对 base_weight 的低秩
        lora_alpha_base=1.0,          # 对 base_weight LoRA 的缩放
        lora_rank_spline=1,           # 对 spline_weight 的低秩(0表示不做LoRA)
        lora_alpha_spline=1.0,        # 对 spline_weight LoRA 的缩放
        # 是否冻结原始权重
        freeze_base_weight=True,
        freeze_spline_weight=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # -------------------
        # 1) 构建网格
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        ) # 得到grid的格子的坐标
        self.register_buffer("grid", grid)

        # -------------------
        # 2) base_weight 相关
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # LoRA for base_weight
        self.lora_rank_base = lora_rank_base
        self.lora_alpha_base = lora_alpha_base
        if self.lora_rank_base > 0:
            self.base_weight_lora_A = torch.nn.Parameter(torch.zeros(self.lora_rank_base, in_features))
            self.base_weight_lora_B = torch.nn.Parameter(torch.zeros(out_features, self.lora_rank_base))
        else:
            self.base_weight_lora_A = None
            self.base_weight_lora_B = None

        # -------------------
        # 3) spline_weight 相关
        # 其形状为 [out_features, in_features, grid_size + spline_order]
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        # 可选地对 spline_weight 也用 LoRA
        self.lora_rank_spline = lora_rank_spline
        self.lora_alpha_spline = lora_alpha_spline
        if self.lora_rank_spline > 0:
            # flatten: out_features x [in_features*(grid_size + spline_order)]
            spline_in_dim = in_features * (grid_size + spline_order)
            self.spline_weight_lora_A = torch.nn.Parameter(torch.zeros(self.lora_rank_spline, spline_in_dim))
            self.spline_weight_lora_B = torch.nn.Parameter(torch.zeros(out_features, self.lora_rank_spline))
        else:
            self.spline_weight_lora_A = None
            self.spline_weight_lora_B = None

        # -------------------
        # 4) spline_scaler 相关
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        else:
            self.register_parameter('spline_scaler', None)

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        # 默认冻结原始权重
        if freeze_base_weight:
            self.base_weight.requires_grad = False
        if freeze_spline_weight:
            self.spline_weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        # base_weight 初始化
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        # spline_weight + random noise
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            # 这里用一个简化的函数 curve2coeff, 但为了演示就不细分
            init_spline = noise.permute(2,1,0).contiguous()  # shape: [out_features, in_features, grid_size+1]
            # 补齐 (grid_size + spline_order)
            spline_shape = list(self.spline_weight.shape)  # [out_features, in_features, grid_size + spline_order]
            self.spline_weight.data.normal_(0, 0.02)
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

        # LoRA 参数初始化
        if self.lora_rank_base > 0:
            # 通常推荐使用较小的标准差
            torch.nn.init.normal_(self.base_weight_lora_A, std=0.02)
            torch.nn.init.normal_(self.base_weight_lora_B, std=0.02)
        if self.lora_rank_spline > 0:
            torch.nn.init.normal_(self.spline_weight_lora_A, std=0.02)
            torch.nn.init.normal_(self.spline_weight_lora_B, std=0.02)

    def b_splines(self, x: torch.Tensor):
        """
        与原 KANLinear 相同, 计算 B-spline bases (batch_size, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        # 初始 one-hot 区间
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1 : (-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()  # (batch_size, in_features, grid_size + spline_order)

    @property
    def effective_base_weight(self):
        """
        W_eff = W + alpha*(B @ A)  (LoRA on base_weight)
        """
        if self.lora_rank_base <= 0:
            return self.base_weight  # 不做 LoRA
        else:
            # B @ A => shape: [out_features, in_features]
            delta = self.base_weight_lora_B @ self.base_weight_lora_A
            return self.base_weight + self.lora_alpha_base * delta

    @property
    def effective_spline_weight(self):
        """
        W_eff_spline = spline_weight + alpha_spline*(B @ A)  (LoRA on spline_weight)
        注意 spline_weight 是 3D，需要先 flatten 之后加完再 unflatten。
        """
        if self.lora_rank_spline <= 0:
            w = self.spline_weight
        else:
            # flatten -> [out_features, in_features*(grid_size+spline_order)]
            # B @ A => shape: [out_features, in_features*(grid_size+spline_order)]
            in_dim = self.in_features * (self.grid_size + self.spline_order)
            delta = self.spline_weight_lora_B @ self.spline_weight_lora_A  # [out_features, in_dim]
            w_flat = self.spline_weight.view(self.out_features, in_dim)
            w_flat = w_flat + self.lora_alpha_spline * delta
            w = w_flat.view(self.out_features, self.in_features, self.grid_size + self.spline_order)

        # 如果有 spline_scaler，则再乘以它
        if self.enable_standalone_scale_spline and hasattr(self, "spline_scaler") and self.spline_scaler is not None:
            # shape of w: [out_features, in_features, grid_size + spline_order]
            # shape of spline_scaler: [out_features, in_features]
            # broadcast在最后一个维度
            w = w * self.spline_scaler.unsqueeze(-1)

        return w

    def forward(self, x: torch.Tensor):
        # (1) 先过 base_weight
        base_output = F.linear(self.base_activation(x), self.effective_base_weight)
        # (2) B-spline 变换
        bs = self.b_splines(x)  # [batch, in_features, grid_size+spline_order]
        bs_flat = bs.view(x.size(0), -1)  # [batch, in_features*(grid_size+spline_order)]

        spline_w = self.effective_spline_weight  # [out_features, in_features, grid_size + spline_order]
        spline_w_flat = spline_w.view(self.out_features, -1)
        spline_output = F.linear(bs_flat, spline_w_flat)

        return base_output + spline_output

    # 下面这些辅助函数照原先 KANLinear 保留或省略:
    def update_grid(self, x: torch.Tensor, margin=0.01):
        pass

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        只示范对 spline_weight 做一下 L1/熵之类的正则；LoRA 部分的正则可自行添加
        """
        with torch.no_grad():
            # 不一定非要算 LoRA 后的 weight
            w = self.spline_weight
        l1_fake = w.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KANLoRA(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        # LoRA 参数（可以统一设，也可以对每层分别设）
        lora_rank_base=4,
        lora_alpha_base=1.0,
        lora_rank_spline=0,
        lora_alpha_spline=1.0,
        freeze_base_weight=True,
        freeze_spline_weight=True,
    ):
        """
        LoRA 版本的 KAN 网络, 多层叠加 KANLinearLoRA。
        """
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinearLoRA(
                    in_features=in_features,
                    out_features=out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    lora_rank_base=lora_rank_base,
                    lora_alpha_base=lora_alpha_base,
                    lora_rank_spline=lora_rank_spline,
                    lora_alpha_spline=lora_alpha_spline,
                    freeze_base_weight=freeze_base_weight,
                    freeze_spline_weight=freeze_spline_weight,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for i, layer in enumerate(self.layers):
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
            # 如果不是最后一层，可以再加非线性，比如 tanh
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
