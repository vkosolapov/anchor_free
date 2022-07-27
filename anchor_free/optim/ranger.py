import math
import torch
from torch.optim.optimizer import Optimizer


def centralized_gradient(x, use_gc=True, gc_conv_only=False):
    if use_gc:
        if gc_conv_only:
            if len(list(x.size())) > 3:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
        else:
            if len(list(x.size())) > 1:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
    return x


class Ranger(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        alpha=0.5,
        k=6,
        N_sma_threshold=5,
        betas=(0.95, 0.999),
        eps=1e-5,
        weight_decay=0,
        use_gc=True,
        gc_conv_only=False,
        gc_loc=True,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        if not lr > 0:
            raise ValueError(f"Invalid Learning Rate: {lr}")
        if not eps > 0:
            raise ValueError(f"Invalid eps: {eps}")
        defaults = dict(
            lr=lr,
            alpha=alpha,
            k=k,
            step_counter=0,
            betas=betas,
            N_sma_threshold=N_sma_threshold,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.N_sma_threshold = N_sma_threshold
        self.alpha = alpha
        self.k = k
        self.radam_buffer = [[None, None, None] for ind in range(10)]
        self.gc_loc = gc_loc
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only

    def __setstate__(self, state):
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Ranger optimizer does not support sparse gradients"
                    )
                p_data_fp32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                    state["slow_buffer"] = torch.empty_like(p.data)
                    state["slow_buffer"].copy_(p.data)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                if self.gc_loc:
                    grad = centralized_gradient(
                        grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only
                    )
                state["step"] += 1
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                buffered = self.radam_buffer[int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshold:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    buffered[2] = step_size
                if N_sma > self.N_sma_threshold:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    G_grad = exp_avg / denom
                else:
                    G_grad = exp_avg
                if group["weight_decay"] != 0:
                    G_grad.add_(p_data_fp32, alpha=group["weight_decay"])
                if self.gc_loc == False:
                    G_grad = centralized_gradient(
                        G_grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only
                    )
                p_data_fp32.add_(G_grad, alpha=-step_size * group["lr"])
                p.data.copy_(p_data_fp32)
                if state["step"] % group["k"] == 0:
                    slow_p = state["slow_buffer"]
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    p.data.copy_(slow_p)
        return loss
