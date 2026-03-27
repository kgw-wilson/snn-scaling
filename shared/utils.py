import torch

def get_pytorch_compile_mode(device: torch.device) -> str:
    if device.type == "cuda":
        compile_mode = "max-autotune"
    else:
        compile_mode = "reduce-overhead"
    return compile_mode