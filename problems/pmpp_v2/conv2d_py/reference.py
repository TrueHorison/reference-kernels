from utils import match_reference, DeterministicContext
import torch
import torch.nn.functional as F
from task import input_t, output_t


def ref_kernel(data: input_t) -> output_t:
    """
    Reference implementation of 2D convolution using PyTorch.
    The sum is done in float64 and rounded to float32, so the reference is the
    true answer to within one float32 rounding, not one cuDNN algorithm's error.
    Args:
        data: Tuple of (input tensor, kernel tensor)
    Returns:
        Output tensor after convolution
    """
    with DeterministicContext():
        input_tensor, kernel, output = data
        return F.conv2d(
            input_tensor.to(torch.float64),
            kernel.to(torch.float64),
            # No padding and no striding
            stride=1,
            padding=0,
        ).to(torch.float32)


def generate_input(
    size: int, kernelsize: int, channels: int, batch: int, seed: int
) -> input_t:
    """
    Generates random input and kernel tensors.
    Returns:
        Tuple of (input tensor, kernel tensor)
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    # Generate input tensor: [batch, in_channels, height, width]
    input_tensor = torch.randn(
        batch, channels, size, size, device="cuda", dtype=torch.float32, generator=gen
    ).contiguous()

    # Generate kernel tensor: [out_channels, in_channels, kernel_height, kernel_width]
    # Here we use same number of output channels as input channels for simplicity
    kernel = torch.randn(
        channels,
        channels,
        kernelsize,
        kernelsize,
        device="cuda",
        dtype=torch.float32,
        generator=gen,
    ).contiguous()

    output_tensor = torch.empty(
        batch,
        channels,
        size - kernelsize + 1,
        size - kernelsize + 1,
        device="cuda",
        dtype=torch.float32,
    )

    return input_tensor, kernel, output_tensor


def check_implementation(data: input_t, output: output_t) -> tuple[bool, str]:
    # Each output is a sum of C * k * k products. For unit normal inputs its
    # size grows like sqrt(C * k * k), and so does the float32 error of a
    # correct kernel (FFT style algorithms are the worst case). A fixed atol is
    # too tight for large reductions. 2e-4 * sqrt(C * k * k) is about 3x the
    # largest plain float32 error measured at C=128, k=32 (2.3e-2 against a
    # float64 result), and still far below the error from TF32, a bad window
    # or a wrong channel.
    _, kernel, _ = data
    reduction = kernel.shape[1] * kernel.shape[2] * kernel.shape[3]
    atol = max(1e-3, 2e-4 * reduction**0.5)
    return match_reference(data, output, reference=ref_kernel, rtol=1e-3, atol=atol)
