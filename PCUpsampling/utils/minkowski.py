import torch.nn.functional as F
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor
from MinkowskiEngine.MinkowskiTensorField import TensorField


def _wrap_tensor(input, F):
    if isinstance(input, TensorField):
        return TensorField(
            F,
            coordinate_field_map_key=input.coordinate_field_map_key,
            coordinate_manager=input.coordinate_manager,
            quantization_mode=input.quantization_mode,
        )
    else:
        return SparseTensor(
            F,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )


def group_norm(input, num_groups, *args, **kwargs):
    return _wrap_tensor(input, F.group_norm(input.F, num_groups=num_groups * args, **kwargs))
