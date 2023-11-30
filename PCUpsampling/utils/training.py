import torch
import numpy as np
import torch.nn.init as init


def to_cuda(data, device):
    if data is None:
        return None
    if isinstance(data, (list, tuple)):
        return [to_cuda(d, device) for d in data]
    if device is None:
        return data.cuda()
    else:
        return data.to(device)


def get_data_batch(batch, cfg, return_dict=False, device=None):
    target = batch["train_points"].transpose(1, 2)

    # load conditioning features
    if not cfg.data.unconditional:
        feature_cond = batch["features"] if "features" in batch else None
        lowres_cond = batch["train_points_lowres"].transpose(1, 2) if "train_points_lowres" in batch else None
    else:
        feature_cond, lowres_cond = None, None

    # move to devices
    target = to_cuda(target, device)
    feature_cond = to_cuda(feature_cond, device)
    lowres_cond = to_cuda(lowres_cond, device)

    if return_dict:
        return {
            "target": target,
            "feature_cond": feature_cond,
            "lowres_cond": lowres_cond,
        }
    return target, feature_cond


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
