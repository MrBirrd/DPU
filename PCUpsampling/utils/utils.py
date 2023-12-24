from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.init as init
from torch import Tensor


def to_cuda(data, device) -> Union[Tensor, List, Tuple, Dict, None]:
    """
    Moves the input data to the specified device (GPU) if available.

    Args:
        data: The input data to be moved to the device.
        device: The device (GPU) to move the data to.

    Returns:
        The input data moved to the specified device.

    """
    if data is None:
        return None
    if isinstance(data, (list, tuple)):
        return [to_cuda(d, device) for d in data]
    if isinstance(data, dict):
        return {k: to_cuda(v, device) for k, v in data.items()}
    if device is None:
        return data.cuda(non_blocking=True)
    else:
        return data.to(device, non_blocking=True)


def ensure_size(x: Tensor) -> Tensor:
    """
    Ensures that the input tensor has the correct size and dimensions.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The input tensor with the correct size and dimensions.
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)
    assert x.dim() == 3
    if x.size(1) > x.size(2):
        x = x.transpose(1, 2)
    return x


def get_data_batch(batch, cfg):
    """
    Get a batch of data for training or testing.

    Args:
        batch (dict): A dictionary containing the batch data.
        cfg (dict): A dictionary containing the configuration settings.

    Returns:
        dict: A dictionary containing the processed batch data.
    """
    hr_points = batch["hr_points"].transpose(1, 2)

    # load conditioning features
    if not cfg.data.unconditional:
        features = batch["features"] if "features" in batch else None
        lr_points = batch["lr_points"] if "lr_points" in batch else None
    else:
        features, lr_points = None, None

    hr_points = ensure_size(hr_points)
    
    features = ensure_size(features) if features is not None else None
    lr_points = ensure_size(lr_points) if lr_points is not None else None
    
    lr_colors = ensure_size(batch["lr_colors"]) if "lr_colors" in batch else None
    hr_colors = ensure_size(batch["hr_colors"]) if "hr_colors" in batch else None

    # concatenate colors to features
    if lr_colors is not None and lr_colors.shape[-1] > 0 and cfg.data.use_rgb_features:
        features = torch.cat([lr_colors, features], dim=1) if features is not None else lr_colors

    # unconditionals training (no features) at all
    if cfg.data.unconditional:
        features = None
        lr_points = None
    
    assert hr_points.shape == lr_points.shape
    
    return {
        "hr_points": hr_points,
        "lr_points": lr_points,
        "features": features,
    }


def smart_load_model_weights(model, pretrained_dict):
    # Get the model's state dict
    model_dict = model.state_dict()

    # New state dict
    new_state_dict = {}
    device = model.device

    for name, param in model_dict.items():
        if name in pretrained_dict:
            # Load the pretrained weight
            pretrained_param = pretrained_dict[name]

            if param.size() == pretrained_param.size():
                # If sizes match, load the pretrained weights as is
                new_state_dict[name] = pretrained_param
            else:
                # Handle size mismatch
                # Resize pretrained_param to match the size of param
                reshaped_param = resize_weight(param.size(), pretrained_param, device=device, layer_name=name)
                new_state_dict[name] = reshaped_param
        else:
            # If no pretrained weight, use the model's original weights
            new_state_dict[name] = param

    # Update the model's state dict
    model.load_state_dict(new_state_dict)


def resize_weight(target_size, weight, layer_name="", device="cpu"):
    """
    Resize the weight tensor to the target size.
    Handles different layer types including attention layers.
    Uses Xavier or He initialization for new weights.
    Args:
        target_size: The desired size of the tensor.
        weight: The original weight tensor.
        layer_name: Name of the layer (used to determine initialization strategy).
        device: The target device ('cpu', 'cuda', etc.)
    """
    # Initialize the target tensor on the specified device
    target_tensor = torch.zeros(target_size, device=device)

    # Copy existing weights
    min_shape = tuple(min(s1, s2) for s1, s2 in zip(target_size, weight.shape))
    slice_objects = tuple(slice(0, min_dim) for min_dim in min_shape)
    target_tensor[slice_objects] = weight[slice_objects].to(device)

    # Mask to identify new weights (those that are still zero)
    mask = (target_tensor == 0).type(torch.float32)

    # Initialize new weights
    if "attention" in layer_name or "conv" in layer_name:
        # He initialization for layers typically followed by ReLU
        new_weights = torch.empty(target_size, device=device)
        init.kaiming_uniform_(new_weights, a=0, mode="fan_in", nonlinearity="relu")
    else:
        # Xavier initialization for other layers
        new_weights = torch.empty(target_size, device=device)
        init.xavier_uniform_(new_weights, gain=init.calculate_gain("linear"))

    # Apply the initialization only to new weights
    target_tensor = target_tensor * (1 - mask) + new_weights * mask

    return target_tensor
