import torch
import torch.nn as nn
from typing import Callable
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import base
from spikingjelly.activation_based import surrogate
from torch.nn.modules.utils import _pair, _single, _triple


# from spikingjelly.activation_based import functional
# from spikingjelly.activation_based.surrogate import SurrogateFunctionBase
# from spikingjelly.clock_driven.surrogate import SurrogateFunctionBase
# from spikingjelly import visualizing
# from matplotlib import pyplot as plt

class SimpleBaseNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode='s'):
        """
        A simple version of ``BaseNode``. The user can modify this neuron easily.
        """
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.detach_reset = detach_reset
        self.step_mode = step_mode
        self.register_memory(name='v', value=0.)
        self.register_memory(name='pre_trace', value=0.)

    def single_step_forward(self, x: torch.Tensor):

        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - self.v_threshold * spike_d

        else:
            # hard reset
            self.v = spike_d * self.v_reset + (1. - spike_d) * self.v


class SimpleIFNode(SimpleBaseNode):
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x


class SimpleLIFNode(SimpleBaseNode):
    def __init__(self, tau: float, decay_input: bool, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode='s'):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.tau = tau
        self.decay_input = decay_input

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            self.v = self.v + (self.v_reset - self.v + x) / self.tau
        else:
            self.v = self.v + (self.v_reset - self.v) / self.tau + x


class SpikeTrace_LIF_Neuron(SimpleLIFNode):
    def __init__(self, layer=None, bn=None, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = None, surrogate_function: Callable = None,
                 detach_reset: bool = False, step_mode='s', **kwargs):

        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.layer = layer
        self.bn = bn
    def neuronal_charge(self, x: torch.Tensor):
        if self.layer is not None:
            # x = self.layer(x)
            if isinstance(self.layer, nn.Conv2d):
                x, self.pre_trace = trace_layer(x, self.pre_trace,
                                             self.tau,
                                             "conv", self.layer.weight, self.layer.bias,
                                             self.layer.stride, self.layer.padding)
            elif isinstance(self.layer, nn.Linear):
                x, self.pre_trace = trace_layer(x, self.pre_trace,
                                             self.tau,
                                             "fc", self.layer.weight, self.layer.bias,
                                             None, None)
        if self.bn is not None:
            x = self.bn(x)

        if self.v_reset is None or self.v_reset == 0:
            if type(self.v) is float:
                self.v = x
            else:
                self.v = self.v.detach() * (1 - 1. / self.tau) + x
        else:
            if type(self.v) is float:
                self.v = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x
            else:
                self.v = self.v.detach() * (1 - 1. / self.tau) + self.v_reset / self.tau + x


class ExponentialSurroGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh):
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        return input.ge(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        thresh = ctx.thresh
        grad_input = grad_output.clone()
        return grad_input * torch.exp(-torch.abs(input)), None


class Nature_exponential(surrogate.SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return ExponentialSurroGrad.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (torch.min((torch.sign(x) + 1), 1.0) - torch.sign(x) * (alpha / 2) * torch.exp(-torch.sign(x) * x))
        # return torch.min(torch.max(1. / alpha * x, 0.5), -0.5)


def conv2d_input(
        input_size,
        weight,
        grad_output,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
):
    r"""Compute the gradient of conv2d with respect to the input of the convolution.

    This is same as the 2D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kH x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::
        # >>> input = torch.randn(1, 1, 3, 3, requires_grad=True)
        # >>> weight = torch.randn(1, 1, 1, 2, requires_grad=True)
        # >>> output = F.conv2d(input, weight)
        # >>> grad_output = torch.randn(output.shape)
        # >>> grad_input = torch.autograd.grad(output, input, grad_output)
        # >>> F.grad.conv2d_input(input.shape, weight, grad_output)
    """
    input = grad_output.new_empty(1).expand(input_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _pair(stride),
        _pair(padding),
        _pair(dilation),
        False,
        [0],
        groups,
        (True, False, False),
    )[0]


def conv2d_weight(
        input,
        weight_size,
        grad_output,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
):
    r"""Compute the gradient of conv2d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iH x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::
        # >>> input = torch.randn(1, 1, 3, 3, requires_grad=True)
        # >>> weight = torch.randn(1, 1, 1, 2, requires_grad=True)
        # >>> output = F.conv2d(input, weight)
        # >>> grad_output = torch.randn(output.shape)
        # >>> # xdoctest: +SKIP
        # >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
        # >>> F.grad.conv2d_weight(input, weight.shape, grad_output)
    """
    weight = grad_output.new_empty(1).expand(weight_size)

    return torch.ops.aten.convolution_backward(
        grad_output,
        input,
        weight,
        None,
        _pair(stride),
        _pair(padding),
        _pair(dilation),
        False,
        [0],
        groups,
        (False, True, False),
    )[1]


class Trace_Layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in_spike, trace_integrate_1, tau,
                layer_type,  weight=None, bias=None, stride=None, padding=None):
        ctx.decay_1 = (1 - 1. / tau)
        ctx.layer_type = layer_type
        tensor_weight = torch.tensor(weight, device=in_spike.device)
        if layer_type == "conv":
            ctx.bias = bias
            ctx.stride = stride
            ctx.padding = padding
            weighted_spike_sum = torch.nn.functional.conv2d(in_spike, tensor_weight, bias, stride, padding)
        elif layer_type == "fc":
            weighted_spike_sum = torch.nn.functional.linear(in_spike, tensor_weight, bias)
        elif layer_type == "direct":
            weighted_spike_sum = in_spike

        trace_integrate_1 = trace_integrate_1 * ctx.decay_1 + in_spike

        ctx.mark_non_differentiable(trace_integrate_1)
        ctx.save_for_backward(tensor_weight, in_spike, trace_integrate_1)

        return weighted_spike_sum, trace_integrate_1

    @staticmethod
    def backward(ctx, grad_weighted_spike_sum, unused_grad_trace_integrate_1):
        weight, in_spike, trace_integrate_1 = ctx.saved_tensors
        trace_integrate = trace_integrate_1
        grad_weighted_spike_sum_clone = grad_weighted_spike_sum.clone()
        if ctx.layer_type == "conv":
            grad_weight = conv2d_weight(trace_integrate, weight.size(), grad_weighted_spike_sum_clone, ctx.stride,
                                        ctx.padding)
            grad_in_spike = conv2d_input(in_spike.size(), weight, grad_weighted_spike_sum_clone, ctx.stride,
                                         ctx.padding)
            # print(grad_in_spike.size())
            # print(grad_weight.size())

            return grad_in_spike, None, None, None, grad_weight, None, None, None  #
        elif ctx.layer_type == "fc":
            trans_trace_integrate = torch.transpose(trace_integrate, dim0=0, dim1=1)
            grad_weight = torch.mm(trans_trace_integrate, grad_weighted_spike_sum_clone)
            grad_in_spike = torch.mm(grad_weighted_spike_sum_clone, weight)
            trans_grad_weight = torch.transpose(grad_weight, dim0=0, dim1=1)
            # print(grad_in_spike)
            # print(trans_grad_weight)
            return grad_in_spike, None, None, None, trans_grad_weight, None, None, None  #
        elif ctx.layer_type == "direct":

            return grad_weighted_spike_sum_clone, None, None, None, None, None, None, None


trace_layer = Trace_Layer.apply
