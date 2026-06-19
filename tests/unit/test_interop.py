import numpy as np
import pytest

from tunix_craftext.interop import (
    ConversionError,
    LoraAdapter,
    ModelTemplate,
    TensorRule,
    convert_state_dict,
    merge_lora_adapters,
)


def test_template_converts_pytorch_linear_layout_to_flax_kernel() -> None:
    template = ModelTemplate(
        name="tiny-linear",
        rules=(
            TensorRule("linear.weight", ("Dense_0", "kernel"), "transpose_2d", (2, 3)),
            TensorRule("linear.bias", ("Dense_0", "bias"), expected_shape=(3,)),
        ),
    )

    params = convert_state_dict(
        {"linear.weight": np.arange(6).reshape(3, 2), "linear.bias": np.array([1, 2, 3])}, template
    )

    np.testing.assert_array_equal(params["Dense_0"]["kernel"], [[0, 2, 4], [1, 3, 5]])
    np.testing.assert_array_equal(params["Dense_0"]["bias"], [1, 2, 3])


def test_template_rejects_unmapped_tensor_in_strict_mode() -> None:
    template = ModelTemplate("tiny", (TensorRule("weight", ("kernel",)),))

    with pytest.raises(ConversionError, match="unexpected"):
        convert_state_dict({"weight": np.zeros((2, 2)), "unsafe": np.ones(1)}, template)


def test_lora_merge_uses_flax_kernel_orientation_without_mutating_base() -> None:
    params = {"Dense_0": {"kernel": np.zeros((2, 3))}}
    adapter = LoraAdapter(
        target=("Dense_0", "kernel"),
        down=np.array([[1.0, 2.0]]),
        up=np.array([[3.0], [4.0], [5.0]]),
        alpha=1.0,
    )

    merged = merge_lora_adapters(params, [adapter])

    np.testing.assert_array_equal(merged["Dense_0"]["kernel"], [[3, 4, 5], [6, 8, 10]])
    np.testing.assert_array_equal(params["Dense_0"]["kernel"], np.zeros((2, 3)))
