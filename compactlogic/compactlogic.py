import torch
import difflogic_cuda
import numpy as np
import convlogic_cuda


########################################################################################################################


class LogicLayer(torch.nn.Module):
    """
    Differentiable logic-gate layer for building logic gate networks.

    Parameters
    ----------
    in_dim : int
        Number of input neurons.
    out_dim : int
        Number of output neurons.
    connections : str
        Input pairing scheme used when sampling connections.
        Supported: {'random', 'original'}.
    num_gates : int, default=128
        Number of candidate gates per output neuron (size of gate set per neuron).
    residual_init : bool, default=False
        If True, initialize weights to favor a residual-like passthrough gate, otherwise gaussian initialization.

    Attributes
    ----------
    weights : torch.nn.Parameter, shape ``(out_dim, num_gates)`` or ``(c_out, num_gates)``
        Learnable logits over gate choices for each output neuron.
    indices : torch.Tensor, shape ``(2, out_dim, num_gates)``
        Integer indices selecting the two input features (a, b) used by each neuron for each gate slot.
    gate_sequence : torch.Tensor, shape ``(out_dim, num_gates)`` or ``(c_out, num_gates)``
        Encodes which logical operation corresponds to each gate slot (uint8 codes).
    is_frozen : bool
        If True, evaluation uses cached argmax gate indices for faster inference.

    Input
    -----
    x : torch.Tensor of shape ``(batch_size, in_dim)``
        Batch of input features.

    Returns
    -------
    y : torch.Tensor of shape ``(batch_size, out_dim)``
        Output features after applying the selected binary logic operations.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            connections: str = 'random',
            num_gates = 16,
            residual_init: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.connections = connections
        self.num_gates = num_gates
        self.log_k = torch.log(torch.tensor(self.num_gates)).item()
        self.init_weights(residual_init)
        self.init_gate_sequence()

        self.get_connections(device=self.weights.device)
        self.refresh_backward_index()

        self.num_neurons = out_dim
        self.num_weights = out_dim

        self.is_frozen = False

    def forward(self, x):
        """
        Run the layer in train or eval mode.
        """
        if getattr(self, "is_frozen", False) or not self.training:
            with torch.no_grad():
                return self.forward_cuda_eval(x)
        return self.forward_cuda(x)

    def forward_cuda(self, x):
        """
        Run the layer in train mode.

        Args:
            x (torch.Tensor): Input tensor with shape ``(batch_size, in_dim)``.

        Returns:
            torch.Tensor: Output tensor with shape ``(batch_size, out_dim)``.

        Notes
        -----
        The CUDA dense kernel expects layout ``(in_dim, batch_size)``, so the Python wrapper
        transposes the logical input before dispatch. See ``compactlogic/cuda/developer.md`` for
        a developer-oriented overview of the kernel structure and tensor conventions.
        """
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1)  # for efficient memory access
        x = x.contiguous()

        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        a, b = self.indices[0], self.indices[1]

        w = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)
        return LogicLayerCudaFunction.apply(
            x,
            a,
            b,
            w,
            self.gate_sequence,
            self.given_x_indices_of_y_start,
            self.given_x_indices_of_y,
            self.given_x_indices_of_gate,
        ).transpose(0, 1)

    @torch.no_grad()
    def forward_cuda_eval(self, x):
        """
        Execute the discrete CUDA inference pass.
        """
        assert self.is_frozen or not self.training
        w = getattr(self, "w", self.weights.argmax(-1).to(dtype=torch.uint8, device=self.weights.device))
        a, b = self.indices[0], self.indices[1]
        x = x.transpose(0, 1).contiguous()
        x = difflogic_cuda.eval(x, a, b, w, self.gate_sequence).transpose(0, 1)
        return x

    def extra_repr(self):
        return '{}, {}, {}'.format(self.in_dim, self.out_dim, 'train' if self.training else 'eval')

    def init_weights(self, residual_init: bool = False):
        """
        Initialize gate-selection logits with random or residual-biased values.
        """
        num_rows = getattr(self, 'c_out', self.out_dim)
        if not residual_init:
            self.weights = torch.nn.parameter.Parameter(torch.randn([num_rows, self.num_gates]))
        else:
            const9 = np.log(9 * (self.num_gates - 1))
            self.weights = torch.zeros((num_rows, self.num_gates))
            residual_gate_idx = 3  # the identity gate
            self.weights[:, residual_gate_idx] = const9
            self.weights = torch.nn.parameter.Parameter(self.weights)

    def init_gate_sequence(self):
        """
        Create and register the per-row sequence of logical gate opcodes.
        """
        num_rows = self.c_out if hasattr(self, "c_out") else self.out_dim
        device = self.weights.device
        base = torch.arange(self.num_gates, device=device, dtype=torch.int64) & 15
        gate_sequence = base.unsqueeze(0).repeat(num_rows, 1).to(torch.uint8)

        self.register_buffer('gate_sequence', gate_sequence)

    def get_connections(self, device='cuda'):
        """
        Sample input index pairs for each output neuron and gate slot. Registers ``indices`` as an internal module
        buffer.
        """
        assert self.out_dim * 2 >= self.in_dim, 'The number of neurons ({}) must not be smaller than half of the ' \
                                                'number of inputs ({}) because otherwise not all inputs could be ' \
                                                'used or considered.'.format(self.out_dim, self.in_dim)
        if self.connections == 'random':
            perm2 = torch.argsort(torch.rand(2 * self.out_dim, self.num_gates, device=device), dim=0)
            c = perm2 % self.in_dim
            perm_in = torch.argsort(torch.rand(self.in_dim, self.num_gates, device=device), dim=0)
            c = torch.gather(perm_in, dim=0, index=c)
            c = c.view(2, self.out_dim, self.num_gates)
        elif self.connections == 'original':  # each output neuron uses one input pairs
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = torch.broadcast_to(c.unsqueeze(-1), (*c.shape, self.num_gates))
            c = c.reshape(2, self.out_dim, self.num_gates)
        else:
            raise NotImplementedError(f'self.connections: "{self.connections}" not implemented.')
        c = c.to(torch.int64).to(device)
        self.register_buffer('indices', c)

    def refresh_backward_index(self, changed_rows: torch.Tensor | None = None) -> None:
        """
        Build inverse connectivity metadata for the regular-layer CUDA backward pass.

        For each input feature, we store the flattened list of (output neuron, gate slot, operand side)
        occurrences that depend on it. This lets the CUDA backward compute ``grad_x`` without atomic adds.
        """
        if not hasattr(self, "indices") or self.indices.ndim != 3:
            return

        device = self.indices.device
        num_inputs = self.in_dim
        num_outputs, num_gates = self.indices.shape[1], self.indices.shape[2]

        gate_ids = torch.arange(num_gates, device=device, dtype=torch.int64)

        def pack_entries(
            input_indices: torch.Tensor,
            row_indices: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            row_indices = row_indices.to(device=device, dtype=torch.int64).flatten()
            if row_indices.numel() == 0:
                empty_i64 = torch.empty(0, dtype=torch.int64, device=device)
                empty_i8 = torch.empty(0, dtype=torch.int8, device=device)
                return empty_i64, empty_i64, empty_i8

            row_gate_ids = gate_ids.unsqueeze(0).expand(row_indices.numel(), -1)
            row_ids = row_indices.unsqueeze(1).expand(-1, num_gates)

            flat_input = torch.cat(
                (
                    input_indices[0, row_indices].reshape(-1),
                    input_indices[1, row_indices].reshape(-1),
                ),
                dim=0,
            )
            flat_y = torch.cat((row_ids.reshape(-1), row_ids.reshape(-1)), dim=0)
            flat_gate = torch.cat((row_gate_ids.reshape(-1), -(row_gate_ids.reshape(-1) + 1)), dim=0).to(torch.int8)
            return flat_input, flat_y, flat_gate

        def finalize_entries(
            flat_input: torch.Tensor,
            flat_y: torch.Tensor,
            flat_gate: torch.Tensor,
        ) -> None:
            order = flat_input.argsort()
            sorted_input = flat_input[order]
            flat_y_tensor = flat_y[order]
            flat_gate_tensor = flat_gate[order]

            counts = torch.bincount(sorted_input, minlength=num_inputs)
            starts_tensor = torch.empty(num_inputs + 1, dtype=torch.int64, device=device)
            starts_tensor[0] = 0
            starts_tensor[1:] = counts.cumsum(0)

            self._set_or_register_buffer("given_x_indices_of_y_start", starts_tensor)
            self._set_or_register_buffer("given_x_indices_of_y", flat_y_tensor)
            self._set_or_register_buffer("given_x_indices_of_gate", flat_gate_tensor)

        if changed_rows is None or "given_x_indices_of_y_start" not in self._buffers:
            all_rows = torch.arange(num_outputs, device=device, dtype=torch.int64)
            flat_input, flat_y, flat_gate = pack_entries(self.indices, all_rows)
            finalize_entries(flat_input, flat_y, flat_gate)
            return

        changed_rows = changed_rows.to(device=device, dtype=torch.int64).flatten()
        if changed_rows.numel() == 0:
            return
        changed_rows = torch.unique(changed_rows)

        row_mask = torch.zeros(num_outputs, dtype=torch.bool, device=device)
        row_mask[changed_rows] = True

        starts = self.given_x_indices_of_y_start
        old_y = self.given_x_indices_of_y
        old_gate = self.given_x_indices_of_gate
        counts = starts[1:] - starts[:-1]
        old_input = torch.arange(num_inputs, device=device, dtype=torch.int64).repeat_interleave(counts)

        keep_mask = ~row_mask[old_y]
        kept_input = old_input[keep_mask]
        kept_y = old_y[keep_mask]
        kept_gate = old_gate[keep_mask]

        new_input, new_y, new_gate = pack_entries(self.indices, changed_rows)

        flat_input = torch.cat((kept_input, new_input), dim=0)
        flat_y = torch.cat((kept_y, new_y), dim=0)
        flat_gate = torch.cat((kept_gate, new_gate), dim=0)

        finalize_entries(flat_input, flat_y, flat_gate)

    def _set_or_register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = tensor
        else:
            self.register_buffer(name, tensor)

    @torch.no_grad()
    def resample_connections(
            self,
            converged_rows,
            threshold_high: float = 0.95,
            threshold_low: float = 0.0,
            update_gate = True,
            keep_strong_weights = True,
    ):
        """
        Recompute weights and connections ONLY for rows in ``converged_rows``
        """
        device = self.weights.device
        num_rows = self.weights.shape[0]

        assert len(converged_rows.shape) == 1
        converged_rows = converged_rows.detach().clone().to(device)

        sel_mask = torch.zeros(num_rows, device=device, dtype=torch.bool)
        sel_mask[converged_rows] = True

        probs = torch.nn.functional.softmax(self.weights, dim=-1)
        row_max, row_argmax = probs.max(dim=1)

        strong_mask_all = row_max > threshold_high
        weak_mask_all = row_max < threshold_low

        strong_mask = strong_mask_all & sel_mask
        weak_mask = weak_mask_all & sel_mask

        n_strong = strong_mask.sum().item()
        n_weak = weak_mask.sum().item()

        gate_pool = torch.arange(16, device=device, dtype=torch.uint8)

        if n_weak:
            self.weights[weak_mask] = torch.randn((n_weak, self.num_gates), device=device)
            if update_gate:
                gate_indices = torch.randint(0, len(gate_pool), (n_weak, self.num_gates), device=device)
                self.gate_sequence[weak_mask] = gate_pool[gate_indices]
            self.update_indices_weak(n_weak, weak_mask)

        if n_strong:
            mask2d = torch.zeros_like(self.weights, dtype=torch.bool, device=device)
            mask2d[strong_mask] = True

            strong_rows = strong_mask.nonzero(as_tuple=False).squeeze(-1)
            keep_cols = row_argmax[strong_rows]
            mask2d[strong_rows, keep_cols] = False

            strong_row_max = row_max[strong_mask].clamp(min=1e-6, max=1 - 1e-6)
            self.weights[strong_mask] = 0.0
            self.weights[strong_rows, keep_cols] = torch.log(
                strong_row_max * (self.num_gates - 1) / (1 - strong_row_max)
            ) if keep_strong_weights else np.log(9 * (self.num_gates - 1))

            num_to_fill = mask2d.sum().item()
            if update_gate and num_to_fill > 0:
                gate_indices = torch.randint(0, len(gate_pool), (num_to_fill,), device=device)
                self.gate_sequence[mask2d] = gate_pool[gate_indices]

            self.update_indices_strong(mask2d, num_to_fill)
        if n_weak or n_strong:
            # Rebuild the full inverse index after rewiring.
            changed_rows = torch.nonzero(torch.logical_or(weak_mask, strong_mask), as_tuple=False).flatten()
            self.refresh_backward_index(changed_rows)
        return weak_mask, strong_mask

    @torch.no_grad()
    def update_indices_weak(self, n_weak, weak_mask):
        """
        Resample both input indices for weak rows selected for rewiring.
        """
        device = self.weights.device
        self.indices[0][weak_mask] = torch.randint(0, self.in_dim, (n_weak, self.num_gates), device=device,
                                                   dtype=torch.int64)
        self.indices[1][weak_mask] = torch.randint(0, self.in_dim, (n_weak, self.num_gates), device=device,
                                                   dtype=torch.int64)

    @torch.no_grad()
    def update_indices_strong(self, mask2d, num_to_fill):
        """
        Resample only non-kept gate slots for strong rows being refreshed.
        """
        device = self.weights.device
        self.indices[0][mask2d] = torch.randint(0, self.in_dim, (num_to_fill,), device=device, dtype=torch.int64)
        self.indices[1][mask2d] = torch.randint(0, self.in_dim, (num_to_fill,), device=device, dtype=torch.int64)

    @torch.no_grad()
    def weights_entropy(self, dim: int = -1) -> torch.Tensor:
        """
        Compute normalized entropy of gate probabilities.

        Args:
            dim (int): Dimension along which the gate distribution is normalized.

        Returns:
            torch.Tensor: Entropy values normalized by ``log(num_gates)``.
        """
        log_p = torch.log_softmax(self.weights, dim=dim)
        p = log_p.exp()

        entropy = -(p * log_p).sum(dim=dim)
        return entropy / self.log_k

    def freeze(self):
        """
        Freeze this layer to deterministic argmax gates for fast inference.
        """
        self.weights.requires_grad = False
        w = self.weights.argmax(-1).to(dtype=torch.uint8, device=self.weights.device)
        self.register_buffer("w", w)
        self.is_frozen = True

    def unfreeze(self):
        self.weights.requires_grad = True
        self.is_frozen = False

    @torch.no_grad()
    def connected_neurons(
        self,
        y_flag: torch.tensor,
        const_gates=torch.tensor([0, 15], device='cuda'),
        trivial_gates_a=torch.tensor([3, 12], device='cuda'),  # a, nota
        trivial_gates_b=torch.tensor([5, 10], device='cuda'),  # b, notb
        device=torch.device('cuda')
    ):
        """
        Compute upstream input-neuron activity masks for pruning.

        Args:
            y_flag (torch.Tensor): Boolean mask of active outputs with shape ``(out_dim,)``.
            const_gates (torch.Tensor): Gate IDs that ignore both inputs.
            trivial_gates_a (torch.Tensor): Gate IDs that depend only on ``a``.
            trivial_gates_b (torch.Tensor): Gate IDs that depend only on ``b``.
            device (torch.device): Device for mask computations.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(x_flag, y_flag)`` boolean masks for
            connected input neurons and remaining non-trivial outputs.
        """
        assert y_flag.shape == (self.out_dim,), f'y_flag shape mismatch: {y_flag.shape}, should be {self.out_dim}'
        x_flag = torch.zeros(self.in_dim, dtype=bool, device=device)
        w = getattr(self, 'w', self.weights.argmax(-1)).to(torch.long)
        indices = torch.gather(self.indices, dim=2, index=w.view(1, -1, 1).expand(2, -1, 1)).squeeze(-1)

        g = torch.gather(self.gate_sequence, dim=1, index=w.unsqueeze(-1)).squeeze(-1)
        g_const_flag = torch.isin(g, const_gates)
        g_trivial_a_flag = torch.isin(g, trivial_gates_a)
        g_trivial_b_flag = torch.isin(g, trivial_gates_b)
        connect_a = (~ g_const_flag) & (~ g_trivial_b_flag)
        connect_b = (~ g_const_flag) & (~ g_trivial_a_flag)

        y_flag[g_const_flag] = False

        connect_indices_a = indices[0][y_flag & connect_a]
        connect_indices_b = indices[1][y_flag & connect_b]
        connect_indices = torch.unique(torch.cat((connect_indices_a, connect_indices_b)))
        x_flag[connect_indices] = True

        y_flag[g_trivial_a_flag | g_trivial_b_flag] = False

        return x_flag, y_flag


########################################################################################################################


class GroupSum(torch.nn.Module):
    """
    The GroupSum module.
    """
    def __init__(self, k: int, tau: float = 1., device='cuda'):
        super().__init__()
        self.k = k
        self.tau = tau
        self.device = device

    def forward(self, x):
        if x.shape[-1] % self.k != 0:
            zero_columns = torch.zeros((x.shape[0], self.k - x.shape[1] % self.k), device=x.device)
            x = torch.cat([x, zero_columns], dim=1)
        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau

    def extra_repr(self):
        return 'k={}, tau={}'.format(self.k, self.tau)


########################################################################################################################


class LogicLayerCudaFunction(torch.autograd.Function):
    """
    Autograd bridge for the dense CUDA kernels.

    Keep this wrapper synchronized with:
    - ``compactlogic/cuda/difflogic.cpp``
    - ``compactlogic/cuda/difflogic_kernel.cu``
    - the tensor-layout notes in ``compactlogic/cuda/developer.md``
    """
    @staticmethod
    def forward(ctx, x, a, b, w, gate_sequence, given_x_indices_of_y_start, given_x_indices_of_y, given_x_indices_of_gate):
        ctx.save_for_backward(x, a, b, w, gate_sequence, given_x_indices_of_y_start, given_x_indices_of_y, given_x_indices_of_gate)
        return difflogic_cuda.forward(x, a, b, w, gate_sequence)

    @staticmethod
    def backward(ctx, grad_y):
        x, a, b, w, gate_sequence, given_x_indices_of_y_start, given_x_indices_of_y, given_x_indices_of_gate = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda.backward_x(
                x,
                a,
                b,
                w,
                gate_sequence,
                grad_y,
                given_x_indices_of_y_start,
                given_x_indices_of_y,
                given_x_indices_of_gate,
            )
        if ctx.needs_input_grad[3]:
            grad_w = difflogic_cuda.backward_w(x, a, b, gate_sequence, grad_y)
        return grad_x, None, None, grad_w, None, None, None, None


########################################################################################################################


class ConvLayer(LogicLayer):
    """
    Convolutional differentiable logic-gate layer (spatial variant of LogicLayer).

    Parameters
    ----------
    in_shape : tuple[int, int, int]
        Input tensor shape as (c_in, h_in, w_in) before padding.
    padding : int
        Zero-padding applied symmetrically to height and width. Internally, effective spatial
        size becomes (h_in + 2*padding, w_in + 2*padding).
    c_out : int
        Number of output channels.
    ks : int
        Kernel size (receptive field size). For ks>1, spatial offsets are sampled in [0, ks).
    stride : int
        Convolution stride used to map output locations back to input coordinates.
    num_chn : int, default=0
        Channel grouping control for sampling offsets:
        - 0: fully random input channel mapping
        - 1: surjective one-to-one channel mapping
        - >1: grouped connectivity; outputs are partitioned into groups that each draw from a
          subset of input channels.
    **kwargs
        Passed to LogicLayer (e.g., input_type, num_gates, device,
        residual_init).

    Attributes
    ----------
    in_shape : tuple[int, int, int]
        Original input shape (c_in, h_in, w_in) without padding.
    out_shape : tuple[int, int, int]
        Output shape (c_out, h_out, w_out), where:
        h_out = (h_in + 2*padding - ks) // stride + 1,
        w_out = (w_in + 2*padding - ks) // stride + 1.
    offsets_ch : torch.Tensor, shape (2, c_out, num_gates), dtype int32
        Input channel indices for the two operands (a, b) per output channel.
    offsets_h : torch.Tensor, shape (2, c_out, num_gates), dtype int8
        Height offsets (relative to the top-left of the receptive field), later shifted by
        `-padding` so coordinates are relative to the unpadded input frame.
    offsets_w : torch.Tensor, shape (2, c_out, num_gates), dtype int8
        Width offsets (relative to the top-left of the receptive field), later shifted by
        `-padding` so coordinates are relative to the unpadded input frame.
    weights : torch.nn.Parameter, shape (c_out, num_gates)
        Learnable logits over gate choices per output channel.
    gate_sequence : torch.Tensor, shape (c_out, num_gates), dtype uint8
        Gate opcode (logic operation) associated with each gate slot.

    Input
    -----
    x : torch.Tensor of shape (batch_size, c_in, h_in, w_in)
        Batch of input feature maps (unpadded). Padding is handled internally by offsets.

    Returns
    -------
    y : torch.Tensor of shape (batch_size, c_out, h_out, w_out)
        Output feature maps after applying the selected binary logic operations.
    """
    def __init__(
        self,
        in_shape: tuple[int, int, int],
        padding: int,
        c_out: int,
        ks: int,
        stride: int,
        num_chn: int = 1,
        **kwargs,
    ):
        self.padding = padding
        self.c_in, self.h_in, self.w_in = in_shape[0], in_shape[1] + 2 * self.padding, in_shape[2] + 2 * self.padding
        self.in_dim = np.prod((self.c_in, self.h_in, self.w_in))
        self.in_shape = in_shape

        self.c_out = c_out
        self.ks = ks
        self.stride = stride
        self.h_out = (self.h_in - self.ks) // self.stride + 1
        self.w_out = (self.w_in - self.ks) // self.stride + 1
        self.out_shape = (self.c_out, self.h_out, self.w_out)
        self.out_dim = np.prod(self.out_shape)
        self.chn_grp = num_chn if num_chn <= 1 else (self.c_in // num_chn)

        super().__init__(self.in_dim, self.out_dim, **kwargs)

    def forward_cuda(self, x):
        """
        Execute convolutional differentiable forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch, c_in, h_in, w_in)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch, c_out, h_out, w_out)``.

        Notes
        -----
        The convolutional CUDA path consumes the NCHW tensor directly together with sampled operand
        offsets. See ``compactlogic/cuda/developer.md`` for the kernel ownership model and layout
        conventions used by the conv implementation.
        """
        assert x.shape[1:] == self.in_shape, f'input shape mismatch {x.shape}, {self.in_shape}'

        if self.training:
            assert x.device.type == 'cuda', x.device
        assert x.ndim == 4, x.ndim

        x = x.contiguous()

        w = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)
        return ConvLayerCudaFunction.apply(
            x, self.offsets_ch, self.offsets_h, self.offsets_w,
            w, self.gate_sequence, self.stride, self.padding, self.ks
        )

    @torch.no_grad()
    def forward_cuda_eval(self, x):
        """
        Execute the discrete CUDA inference pass.
        """
        assert self.is_frozen or not self.training
        w = getattr(self, "w", self.weights.argmax(-1).to(dtype=torch.uint8, device=self.weights.device))
        x = convlogic_cuda.eval(
            x, self.offsets_ch, self.offsets_h, self.offsets_w,
            w, self.gate_sequence, self.stride, self.padding, self.ks
        )

        return x

    def get_connections(self, device='cuda'):
        """
        Sample channel and spatial offsets used by each gate.

        Args:
            device (str): Device used to create sampled offsets.

        Returns:
            None: Registers ``offsets_ch``, ``offsets_h``, and ``offsets_w`` buffers.
        """
        if self.chn_grp > 1:
            assert self.chn_grp <= self.c_in and self.chn_grp <= self.c_out
            offsets_h = torch.randint(0, self.ks, size=(2, self.c_out, self.num_gates), device=device, dtype=torch.int8)
            offsets_w = torch.randint(0, self.ks, size=(2, self.c_out, self.num_gates), device=device, dtype=torch.int8)

            in_channels_per_group = self.c_in // self.chn_grp
            remainder = self.c_in % self.chn_grp
            group_sizes = torch.full((self.chn_grp,), in_channels_per_group, device=device)
            group_sizes[:remainder] += 1
            starts = torch.cat([torch.zeros(1, device=device), group_sizes.cumsum(0)]).to(torch.int32)
            all_input_channels = torch.arange(self.c_in, device=device, dtype=torch.int32)
            group_map = [all_input_channels[starts[i] : starts[i+1]] for i in range(len(starts) - 1)]
            out_channels_per_group = self.c_out // self.chn_grp
            group_ids = torch.clamp(
                torch.arange(self.c_out, device=device, dtype=torch.int32) // out_channels_per_group,
                0, self.chn_grp - 1
            )
            offsets_ch = torch.empty((2, self.c_out, self.num_gates), device=device, dtype=torch.int32)
            for group_id, channels in enumerate(group_map):
                mask = group_ids == group_id
                idxs = mask.nonzero(as_tuple=True)[0]
                cnt = idxs.numel()
                if cnt > 0:
                    k = channels.numel()
                    offsets_ch[0, idxs] = channels[torch.randint(0, k, (cnt, self.num_gates), device=device)]
                    offsets_ch[1, idxs] = channels[torch.randint(0, k, (cnt, self.num_gates), device=device)]
        elif self.chn_grp == 1:
            assert self.ks > 1
            offsets_ch = torch.arange(0, self.c_out, device=device, dtype=torch.int32) % self.c_in
            offsets_ch = offsets_ch.unsqueeze(-1).expand(2, -1, self.num_gates)
            offsets_h = torch.randint(0, self.ks, size=(2, self.c_out, self.num_gates), device=device, dtype=torch.int8)
            offsets_w = torch.randint(0, self.ks, size=(2, self.c_out, self.num_gates), device=device, dtype=torch.int8)
        else:
            offsets_ch = torch.randint(0, self.c_in, size=(2, self.c_out, self.num_gates), device=device, dtype=torch.int32)
            offsets_h = torch.randint(0, self.ks, size=(2, self.c_out, self.num_gates), device=device, dtype=torch.int8)
            offsets_w = torch.randint(0, self.ks, size=(2, self.c_out, self.num_gates), device=device, dtype=torch.int8)

        offsets_ch, offsets_h, offsets_w = self.resolve_conflicts(offsets_ch, offsets_h, offsets_w, self.ks, self.c_in)
        offsets_h -= self.padding
        offsets_w -= self.padding
        offsets_ch = offsets_ch.contiguous().to(device)
        offsets_h = offsets_h.contiguous().to(device)
        offsets_w = offsets_w.contiguous().to(device)

        self.register_buffer("offsets_ch", offsets_ch)
        self.register_buffer("offsets_h", offsets_h)
        self.register_buffer("offsets_w", offsets_w)

    @staticmethod
    def resolve_conflicts(offsets_ch, offsets_h, offsets_w, ks, c_in):
        """
        Resolve cases where a gate sampled identical operands.

        Args:
            offsets_ch (torch.Tensor): Channel offsets for both operands.
            offsets_h (torch.Tensor): Height offsets for both operands.
            offsets_w (torch.Tensor): Width offsets for both operands.
            ks (int): Kernel size.
            c_in (int): Number of input channels.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Conflict-free
            ``(offsets_ch, offsets_h, offsets_w)``.
        """
        conflict = (
                (offsets_ch[0] == offsets_ch[1]) &
                (offsets_h[0] == offsets_h[1]) &
                (offsets_w[0] == offsets_w[1])
        )

        while conflict.any():
            num = conflict.sum()

            if ks > 1:
                offsets_h[1][conflict] = torch.randint(
                    0, ks, size=(num,), device=offsets_h.device, dtype=torch.int8
                )
                offsets_w[1][conflict] = torch.randint(
                    0, ks, size=(num,), device=offsets_w.device, dtype=torch.int8
                )
            elif ks == 1:
                offsets_ch[1][conflict] = torch.randint(
                    0, c_in, size=(num,), device=offsets_ch.device, dtype=torch.int32
                )

            conflict = (
                    (offsets_ch[0] == offsets_ch[1]) &
                    (offsets_h[0] == offsets_h[1]) &
                    (offsets_w[0] == offsets_w[1])
            )

        return offsets_ch, offsets_h, offsets_w

    @torch.no_grad()
    def update_indices_weak(self, n_weak, weak_mask):
        """
        Resample weak convolutional offsets for selected rows.
        """
        device = self.weights.device
        if self.ks == 1:
            self.offsets_ch[0][weak_mask] = torch.randint(0, self.c_in, size=(n_weak, self.num_gates), device=device,
                                                          dtype=torch.int32)
            self.offsets_ch[1][weak_mask] = torch.randint(0, self.c_in, size=(n_weak, self.num_gates), device=device,
                                                          dtype=torch.int32)
        else:
            self.offsets_h[0][weak_mask] = torch.randint(0, self.ks, size=(n_weak, self.num_gates), device=device,
                                                         dtype=torch.int8) - self.padding
            self.offsets_h[1][weak_mask] = torch.randint(0, self.ks, size=(n_weak, self.num_gates), device=device,
                                                         dtype=torch.int8) - self.padding
            self.offsets_w[0][weak_mask] = torch.randint(0, self.ks, size=(n_weak, self.num_gates), device=device,
                                                         dtype=torch.int8) - self.padding
            self.offsets_w[1][weak_mask] = torch.randint(0, self.ks, size=(n_weak, self.num_gates), device=device,
                                                         dtype=torch.int8) - self.padding

    @torch.no_grad()
    def update_indices_strong(self, mask2d, num_to_fill):
        """
        Resample selected convolutional offset entries for strong rows.
        """
        device = self.weights.device
        if self.ks == 1:
            self.offsets_ch[0][mask2d] = torch.randint(0, self.c_in, size=(num_to_fill,), device=device, dtype=torch.int32)
            self.offsets_ch[1][mask2d] = torch.randint(0, self.c_in, size=(num_to_fill,), device=device, dtype=torch.int32)
        else:
            self.offsets_h[0][mask2d] = torch.randint(0, self.ks, size=(num_to_fill,), device=device, dtype=torch.int8) - self.padding
            self.offsets_h[1][mask2d] = torch.randint(0, self.ks, size=(num_to_fill,), device=device, dtype=torch.int8) - self.padding
            self.offsets_w[0][mask2d] = torch.randint(0, self.ks, size=(num_to_fill,), device=device, dtype=torch.int8) - self.padding
            self.offsets_w[1][mask2d] = torch.randint(0, self.ks, size=(num_to_fill,), device=device, dtype=torch.int8) - self.padding

    def extra_repr(self):
        return 'in_shape={}, out_shape={}, pad={}, kernel_size={}, stride={}, num_chn={}, {}'.format(
            self.in_shape,
            self.out_shape,
            self.padding,
            self.ks,
            self.stride,
            self.chn_grp if self.chn_grp <= 1 else (self.c_in // self.chn_grp),
            'train' if self.training else 'eval',
            'frozen' if self.is_frozen else 'unfrozen'
        )

    @torch.no_grad()
    def connected_neurons(
        self,
        y_flag: torch.tensor,
        const_gates = torch.tensor([0, 15], device='cuda'),
        trivial_gates = torch.tensor([3, 5, 10, 12], device='cuda'),
        device=torch.device('cuda')
    ):
        """Generate the mask of connected neurons for pruning"""

        assert y_flag.shape == self.out_shape, f'y_flag shape mismatch: {y_flag.shape}, should be {self.out_shape}'
        x_flag = torch.zeros(self.in_shape, dtype=bool, device=device)
        w = getattr(self, 'w', self.weights.argmax(-1)).to(torch.long)
        offsets_ch = torch.gather(self.offsets_ch, dim=2, index=w.view(1, -1, 1).expand(2, -1, 1)).squeeze(-1)
        offsets_h = torch.gather(self.offsets_h, dim=2, index=w.view(1, -1, 1).expand(2, -1, 1)).squeeze(-1)
        offsets_w = torch.gather(self.offsets_w, dim=2, index=w.view(1, -1, 1).expand(2, -1, 1)).squeeze(-1)

        g = torch.gather(self.gate_sequence, dim=1, index=w.unsqueeze(-1)).squeeze(-1)
        g_const_flag = torch.isin(g, const_gates)
        g_trivial_flag = torch.isin(g, trivial_gates)
        y_flag[g_const_flag] = False

        for c_out in torch.nonzero(~ g_const_flag, as_tuple=True)[0]:
            gate_type = g[c_out]
            update_a = False if gate_type == 5 or gate_type == 10 else True
            update_b = False if gate_type == 3 or gate_type == 12 else True
            for coord in torch.nonzero(y_flag[c_out]):
                if update_a:
                    h_0 = coord[0] * self.stride + offsets_h[0][c_out]
                    w_0 = coord[1] * self.stride + offsets_w[0][c_out]
                    if 0 <= h_0 < self.in_shape[1] and 0 <= w_0 < self.in_shape[2]:
                        x_flag[offsets_ch[0][c_out], h_0, w_0] |= True
                if update_b:
                    h_1 = coord[0] * self.stride + offsets_h[1][c_out]
                    w_1 = coord[1] * self.stride + offsets_w[1][c_out]
                    if 0 <= h_1 < self.in_shape[1] and 0 <= w_1 < self.in_shape[2]:
                        x_flag[offsets_ch[1][c_out], h_1, w_1] |= True

        y_flag[g_trivial_flag] = False
        return x_flag, y_flag


########################################################################################################################


class ConvLayerCudaFunction(torch.autograd.Function):
    """
    Autograd bridge for the convolutional CUDA kernels.

    Keep this wrapper synchronized with:
    - ``compactlogic/cuda/convlogic.cpp``
    - ``compactlogic/cuda/convlogic_kernel.cu``
    - the design notes in ``compactlogic/cuda/developer.md``
    """
    @staticmethod
    def forward(ctx, x, offsets_ch, offsets_h, offsets_w, w, gate_sequence, stride, pad, ks):
        y = convlogic_cuda.forward(x, offsets_ch, offsets_h, offsets_w, w, gate_sequence, stride, pad, ks)
        ctx.save_for_backward(x, offsets_ch, offsets_h, offsets_w, w, gate_sequence)
        ctx.stride = stride
        ctx.pad = pad
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, offsets_ch, offsets_h, offsets_w, w, gate_sequence = ctx.saved_tensors
        stride = ctx.stride
        pad = ctx.pad
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = convlogic_cuda.backward_x(x, offsets_ch, offsets_h, offsets_w, w, gate_sequence, grad_y, stride, pad)
        if ctx.needs_input_grad[4]:
            grad_w = convlogic_cuda.backward_w(x, offsets_ch, offsets_h, offsets_w, gate_sequence, grad_y, stride)
        return grad_x, None, None, None, grad_w, None, None, None, None


########################################################################################################################


class SequentialEntropyFreezer:
    """
    Sequentially freeze logic layers when their entropy dynamics converge.
    """
    def __init__(
        self,
        model,
        ema_decay=0.99,
        eps=5e-4,
        patience=200,
        min_iters=0,
        l=None,
    ):
        """
        Initialize sequential entropy-based freezing.

        Parameters
        ----------
        model : torch.nn.Module or Sequence[torch.nn.Module]
            Container that is iterated to discover candidate layers. Layers with a ``freeze`` attribute are considered
            logic layers.
        ema_decay : float, default=0.99
            Decay factor for the EMA of entropy values.
        eps : float, default=5e-4
            Convergence tolerance for ``abs(current_entropy - previous_ema)``.
        patience : int, default=200
            Number of consecutive converged steps required before freezing.
        min_iters : int, default=0
            Minimum global iteration before convergence checks begin.
        l : int | None, default=None
            Optional cap on the number of discovered logic layers to manage.
            If ``None``, all logic layers except the last are used.
        """
        self.model = model
        self.ema_decay = ema_decay
        self.eps = eps
        self.patience = patience
        self.min_iters = min_iters
        self.l = l

        all_logic_layers = [
            (i, layer)
            for i, layer in enumerate(model)
            if hasattr(layer, "freeze")
        ]
        if l is None:
            self.logic_layers = all_logic_layers[: -1]
        else:
            self.logic_layers = all_logic_layers[: int(l)]

        self.ema = {}
        self.counter = {}
        self.iter = 0

    def step(self, iter: int):
        """
        Update entropy statistics and freeze the next eligible converged layer.
        """
        self.iter = iter
        if self.iter < self.min_iters:
            return

        for idx, layer in self.logic_layers:
            # enforce sequential freezing
            if not self._can_consider_layer(idx):
                continue

            if layer.is_frozen:
                continue

            h = layer.weights_entropy().mean().item()

            if idx not in self.ema:
                self.ema[idx] = h
                self.counter[idx] = 0
                return

            old = self.ema[idx]
            new = self.ema_decay * old + (1 - self.ema_decay) * h
            self.ema[idx] = new

            if abs(h - old) < self.eps:
                self.counter[idx] += 1
            else:
                self.counter[idx] = 0

            if self.counter[idx] >= self.patience:
                self._freeze_layer(layer)
                break  # freeze only ONE layer per iteration

    def _can_consider_layer(self, idx):
        for i, layer in self.logic_layers:
            if i < idx and not layer.is_frozen:
                return False
        return True

    def _freeze_layer(self, layer):
        layer.freeze()

    def __str__(self):
        return 'ema_decay={}, eps={}, patience={}, min_iters={}'.format(
            self.ema_decay,
            self.eps,
            self.patience,
            self.min_iters,
        )


########################################################################################################################


class FixedSequentialEntropyFreezer:
    """
    Freeze logic layers sequentially according to a fixed step schedule.
    """
    def __init__(
        self,
        model,
        N: int,
        min_iters: int = 0,
        l=None,
    ):
        """
        Initialize fixed-schedule sequential freezing.

        Parameters
        ----------
        model : torch.nn.Module or Sequence[torch.nn.Module]
            Container that is iterated to discover candidate layers. Layers with a ``freeze`` attribute are considered
            logic layers.
        N : int
            Step interval between consecutive layer freezes.
            The first managed layer freezes at ``min_iters + N``, second at ``min_iters + 2*N``, etc.
        min_iters : int, default=0
            Global step offset before scheduling starts.
        l : int | None, default=None
            Optional cap on the number of discovered logic layers to manage.
            If ``None``, all logic layers except the last are used.
        """
        if N <= 0:
            raise ValueError(f'N must be > 0, got {N}.')

        self.model = model
        self.N = int(N)
        self.min_iters = min_iters
        self.l = l

        all_logic_layers = [
            (i, layer)
            for i, layer in enumerate(model)
            if hasattr(layer, "freeze")
        ]
        if l is None:
            self.logic_layers = all_logic_layers[: -1]
        else:
            self.logic_layers = all_logic_layers[: int(l)]

        self.iter = 0

    def step(self, iter: int):
        """
        Freeze at most one eligible layer according to a fixed schedule.
        """
        self.iter = iter

        for order, (_, layer) in enumerate(self.logic_layers, start=1):
            if layer.is_frozen:
                continue

            freeze_at = self.min_iters + order * self.N
            if self.iter >= freeze_at:
                layer.freeze()
            break

    def __str__(self):
        return 'N={}, min_iters={}'.format(
            self.N,
            self.min_iters,
        )


########################################################################################################################


class EntropyRowResampler:
    """
    Row-wise entropy convergence monitor with targeted connection resampling.
    """
    def __init__(
        self,
        model,
        ema_decay: float = 0.99,
        eps: float = 5e-4,
        patience: int = 200,
        min_iters: int = 0,
        max_iters = np.inf,
        only_unfrozen: bool = True,
        **kwargs
    ):
        """
        Initialize row-level entropy resampling.

        Parameters
        ----------
        model : Sequence[torch.nn.Module]
            Container iterated to discover layers with ``resample_connections`` support.
        ema_decay : float, default=0.99
            Decay factor used for EMA updates of per-row entropy.
        eps : float, default=5e-4
            Per-row convergence threshold for entropy change magnitude.
        patience : int, default=200
            Number of consecutive converged updates required before resampling a row.
        min_iters : int, default=0
            Minimum iteration at which resampling checks start.
        max_iters : int | float, default=np.inf
            Maximum iteration at which resampling checks are still applied.
        only_unfrozen : bool, default=True
            If ``True``, skip layers that expose ``is_frozen`` and are frozen.
        **kwargs
            Extra keyword arguments forwarded to each layer's ``resample_connections`` call (e.g., thresholds or gate
            update options).
        """
        self.model = model
        self.ema_decay = ema_decay
        self.eps = eps
        self.patience = patience
        self.min_iters = min_iters
        self.max_iters = max_iters
        self.only_unfrozen = only_unfrozen
        self.resample_kwargs = kwargs

        self.layers = [
            (i, layer)
            for i, layer in enumerate(model) if hasattr(layer, "resample_connections")
        ]

        self.ema = {}
        self.counter = {}
        self.iter = 0

    @torch.no_grad()
    def step(self, iter: int, optimizer):
        self.iter = iter
        if self.iter < self.min_iters or self.iter > self.max_iters:
            return

        for idx, layer in self.layers:
            if self.only_unfrozen and hasattr(layer, "is_frozen") and layer.is_frozen:
                continue

            h = layer.weights_entropy()
            h = h.detach()
            assert h.dim() == 1

            if idx not in self.ema:
                self.ema[idx] = h.clone()
                self.counter[idx] = torch.zeros_like(h, dtype=torch.int32, device=h.device)
                continue

            old = self.ema[idx]
            new = self.ema_decay * old + (1.0 - self.ema_decay) * h
            self.ema[idx] = new

            converged_now = (h - old).abs() < self.eps

            cnt = self.counter[idx]
            cnt = torch.where(converged_now, cnt + 1, torch.zeros_like(cnt))
            self.counter[idx] = cnt

            converged_rows = (cnt >= self.patience).nonzero(as_tuple=False).flatten()

            if converged_rows.numel() > 0:
                weak_mask, strong_mask = layer.resample_connections(converged_rows, **self.resample_kwargs)
                mask = torch.logical_or(weak_mask, strong_mask)
                mask2d = mask[:, None].expand_as(layer.weights)
                self.reset_adam_state_for_mask(optimizer, layer.weights, mask2d)

                self.counter[idx][converged_rows] = 0

    @torch.no_grad()
    def reset_adam_state_for_mask(self, optimizer, param, mask):
        """
        Reset Adam moment buffers for masked parameter entries.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer containing state dictionaries keyed by parameter.
        param : torch.nn.Parameter
            Parameter whose selected entries were resampled.
        mask : torch.Tensor
            Boolean tensor broadcastable to ``param.shape`` indicating entries whose optimizer moments should be zeroed.

        Returns
        -------
        None
            Modifies optimizer state in place when present.
        """
        st = optimizer.state.get(param, None)
        if st is None or len(st) == 0:
            return

        if "exp_avg" in st:
            st["exp_avg"].masked_fill_(mask, 0.0)
        if "exp_avg_sq" in st:
            st["exp_avg_sq"].masked_fill_(mask, 0.0)
        if "max_exp_avg_sq" in st:  # generally not used in this project
            st["max_exp_avg_sq"].masked_fill_(mask, 0.0)

    def __str__(self):
        return 'ema_decay={}, eps={}, patience={}, min_iters={}, max_iters={}, resample_kwargs={}'.format(
            self.ema_decay,
            self.eps,
            self.patience,
            self.min_iters,
            self.max_iters,
            self.resample_kwargs,
        )


########################################################################################################################
