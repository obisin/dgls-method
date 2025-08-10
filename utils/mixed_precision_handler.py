import torch.nn.utils
from typing import Union, List, Dict, Any, Optional
import bitsandbytes as bnb
import torch

"""
Universal Mixed Precision Handler
=================================

Drop-in utility to handle gradient operations with mixed precision (float8, bfloat16, float32)
across any model architecture. Patches PyTorch's gradient functions to handle unsupported
dtype operations gracefully.

Usage:
    from utils.mixed_precision_handler import patch_mixed_precision
    patch_mixed_precision()

    # or with user control:
    patch_mixed_precision(mixed_precision_target='f32')
"""
class CastingHandler:
    def __init__(self):
        self.original_clip_grad_norm = None
        self.dtype_conversion_map = {
            torch.float8_e4m3fn: torch.bfloat16,
            torch.float8_e5m2: torch.bfloat16,
        }
        self.safe_dtypes = {torch.float32, torch.float16, torch.bfloat16}
        self.is_patched = False

    def detect_model_highest_precision(self, layers):
        """Find the single highest precision dtype in the model"""
        all_dtypes = set()
        for layer in layers:
            for param in layer.parameters():
                all_dtypes.add(param.dtype)

        # Precision hierarchy (highest to lowest)
        precision_order = [torch.float64, torch.float32, torch.bfloat16, torch.float16,
                           getattr(torch, 'float8_e4m3fn', None)]

        # Find the highest precision actually used
        for dtype in precision_order:
            if dtype and dtype in all_dtypes:
                return dtype
        return torch.float32  # fallback


    def cast_to_dtype(self, param, target_device, cast_target=None):
        """Cast specific source dtype to target dtype"""
        source_dtype = param.dtype

        if cast_target:
            cast_from, cast_to = cast_target

            # Map string to torch dtypes
            dtype_map = {
                'f32': torch.float32,
                'bf16': torch.bfloat16,
                'f16': torch.float16,
                'f8_e4m3': torch.float8_e4m3fn,
                'f8_e5m2': torch.float8_e5m2,
                'nf4': 'nf4',  # Special handling needed NOT TESTED
                'fp4': 'fp4'  # Special handling needed
            }

            # Only cast if source matches the FROM dtype
            if source_dtype == dtype_map[cast_from]:
                # Handle 4-bit quantization
                if cast_to in ['nf4', 'fp4']:
                    try:
                        import bitsandbytes as bnb
                        quantized, quant_state = bnb.functional.quantize_4bit(
                            param.data, blocksize=64, quant_type=cast_to
                        )
                        # Return dequantized version on target device
                        return bnb.functional.dequantize_4bit(
                            quantized.to(target_device), quant_state.to(target_device)
                        )
                    except ImportError:
                        raise RuntimeError(
                            " bitsandbytes required for 4-bit quantization. Install with: pip install bitsandbytes")
                    except Exception as e:
                        raise RuntimeError(f" 4-bit quantization failed: {e}")
                else:
                    # Regular dtype casting
                    return param.to(target_device, dtype=dtype_map[cast_to], non_blocking=True)

        # No casting needed
        return param.to(target_device, non_blocking=True)

    def replace_with_4bit_layer(self, layer):
        """Replace torch.nn.Linear with bitsandbytes Linear4bit, skip LoRA layers"""
        try:
            # import bitsandbytes as bnb

            for name, module in layer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Skip LoRA layers - they cause dropout issues with 4-bit
                    if any(lora_term in name.lower() for lora_term in ['lora', 'adapter']):
                        # print(f"   Skipping LoRA layer: {name}")
                        continue

                    parent_names = name.split('.')
                    if any(any(lora_term in part.lower() for lora_term in ['lora', 'adapter'])
                           for part in parent_names):
                        continue

                    print(f"   Converting layer: {name}")
                    linear_4bit = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        compute_dtype=torch.float16,
                        quant_type='nf4'
                    )

                    linear_4bit.load_state_dict(module.state_dict())
                    linear_4bit = linear_4bit.to('cuda')

                    # Replace in parent module
                    parent_name = '.'.join(name.split('.')[:-1])
                    module_name = name.split('.')[-1]
                    parent = layer
                    for part in parent_name.split('.'):
                        if part:
                            parent = getattr(parent, part)
                    setattr(parent, module_name, linear_4bit)

            return layer
        except Exception as e:
            print(f"️ 4-bit replacement failed: {e}")
            return layer


    def precast_all_layers(self, cpu_swappable_layers, layers):
        """Cast all layer dtypes once at startup using existing cast_to_dtype"""
        print(" Pre-casting layer dtypes (one-time setup)...")
        casted_layers = 0

        for i, layer in enumerate(layers):
            if i in cpu_swappable_layers:  # Only cast swappable layers
                layer_casted = False

                for name, param in layer.named_parameters():
                    # Use your existing function but stay on current device
                    new_param = self.cast_to_dtype(param, param.device)  # Keep on same device
                    if new_param.dtype != param.dtype:
                        param.data = new_param.data
                        layer_casted = True

                if layer_casted:
                    casted_layers += 1

        print(f"✓ Pre-casted {casted_layers} layers using existing cast_to_dtype function")

# =============================================================================
# PRECISION FRAMEWORK CODE
# =============================================================================


class MixedPrecisionHandler:
    def __init__(self, mixed_precision_target='auto'):
        """
        Args:
            mixed_precision_target: 'auto', 'f32', 'bf16', 'f16'
        """
        self.mixed_precision_target = mixed_precision_target

        # Expanded problematic dtypes that break PyTorch gradient ops
        self.problematic_dtypes = {
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.int8,
            torch.uint8,
            # torch.quint4x2,  # Q4 - commented out, may be added later
        }

        self.safe_dtypes = {torch.float32, torch.bfloat16, torch.float16}

    def get_conversion_target(self, original_dtype):
        """Determine what dtype to convert problematic dtypes to"""

        # User specified target takes precedence
        if self.mixed_precision_target and self.mixed_precision_target != 'auto':
            return self._parse_dtype_string(self.mixed_precision_target)

        # Auto-detection: smart fallback based on original dtype
        if self.mixed_precision_target == 'auto':
            return self._auto_select_target(original_dtype)

        return original_dtype

    def _auto_select_target(self, problematic_dtype):
        """Smart automatic target selection for problematic dtypes"""
        conversion_map = {
            torch.float8_e4m3fn: torch.bfloat16,
            torch.float8_e5m2: torch.bfloat16,
            torch.int8: torch.float16,
            torch.uint8: torch.float16,
            # torch.quint4x2: torch.float16,      # Q4 - commented out
        }
        return conversion_map.get(problematic_dtype, torch.bfloat16)  # fallback

    def _parse_dtype_string(self, dtype_str):
        """Parse string dtype to torch dtype"""
        dtype_map = {
            'f32': torch.float32,
            'bf16': torch.bfloat16,
            'f16': torch.float16,
            'f8_e4m3': torch.float8_e4m3fn,
            'f8_e5m2': torch.float8_e5m2,
            # Q4 options - commented out for now
            # 'nf4': 'nf4',  # Special bitsandbytes handling needed
            # 'fp4': 'fp4',  # Special bitsandbytes handling needed
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    def convert_tensor_for_computation(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.dtype]:
        """Convert tensor to safe dtype for computation"""
        original_dtype = tensor.dtype

        # If already safe, no conversion needed
        if tensor.dtype in self.safe_dtypes:
            return tensor, original_dtype

        # If not problematic but also not in safe list, keep as-is
        if tensor.dtype not in self.problematic_dtypes:
            return tensor, original_dtype

        # Convert problematic dtype to target
        target_dtype = self.get_conversion_target(tensor.dtype)
        converted_tensor = tensor.to(target_dtype)

        return converted_tensor, original_dtype

    def convert_parameters_for_computation(self, parameters: List[torch.nn.Parameter]) -> tuple[
        List[torch.nn.Parameter], Dict[int, torch.dtype]]:
        """Convert parameter gradients to safe dtypes for computation"""
        converted_params = []
        original_dtypes = {}

        for i, param in enumerate(parameters):
            if param.grad is not None:
                converted_grad, original_dtype = self.convert_tensor_for_computation(param.grad)

                # Create new parameter with matching dtype to avoid assignment errors
                safe_param_dtype = converted_grad.dtype
                new_param = torch.nn.Parameter(param.data.to(safe_param_dtype))
                new_param.grad = converted_grad
                converted_params.append(new_param)

                # Store original dtype if conversion happened
                if original_dtype != converted_grad.dtype:
                    original_dtypes[i] = original_dtype
            else:
                converted_params.append(param)

        return converted_params, original_dtypes

    def create_safe_clip_function(self, original_clip_grad_norm):
        """Create the patched gradient clipping function"""
        handler = self  # Capture self in closure

        def safe_clip_grad_norm_(parameters: Union[torch.Tensor, List[torch.Tensor]],
                                 max_norm: float,
                                 norm_type: float = 2.0,
                                 error_if_nonfinite: bool = False,
                                 foreach: Optional[bool] = None) -> torch.Tensor:
            """Safe gradient clipping that handles mixed precision tensors"""

            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]

            # Convert parameters to safe dtypes
            converted_params, original_dtypes = handler.convert_parameters_for_computation(list(parameters))

            # Perform clipping on converted parameters
            total_norm = original_clip_grad_norm(
                converted_params, max_norm,
                norm_type=norm_type,
                error_if_nonfinite=error_if_nonfinite,
                foreach=foreach
            )

            # Copy clipped gradients back to original parameters
            for i, (orig_param, conv_param) in enumerate(zip(parameters, converted_params)):
                if orig_param.grad is not None and conv_param.grad is not None:
                    # Convert back to original dtype if it was changed
                    if i in original_dtypes:
                        orig_param.grad.data.copy_(conv_param.grad.data.to(original_dtypes[i]))
                    else:
                        orig_param.grad.data.copy_(conv_param.grad.data)

            return total_norm

        return safe_clip_grad_norm_


def safe_mixed_device_clip_grad_norm(parameters, max_norm, handler=None):
    """
    Enhanced gradient clipping for mixed devices AND mixed precision.
    Handles CPU/GPU device mismatches and dtype conversions.
    """
    # Use global handler if none provided
    if handler is None:
        handler = _GLOBAL_HANDLER

    # Separate parameters by device
    cpu_params = []
    gpu_params = []

    for param in parameters:
        if param.grad is not None:
            if param.grad.device.type == 'cpu':
                cpu_params.append(param)
            else:
                gpu_params.append(param)

    # Clip each device group separately using safe clipping
    total_norm = 0.0

    safe_clip_fn = handler.create_safe_clip_function(torch.nn.utils.clip_grad_norm_)

    if cpu_params:
        cpu_norm = safe_clip_fn(cpu_params, max_norm, norm_type=2.0)
        total_norm += cpu_norm.cpu().item() ** 2

    if gpu_params:
        gpu_norm = safe_clip_fn(gpu_params, max_norm, norm_type=2.0)
        total_norm += gpu_norm.cpu().item() ** 2

    return (total_norm ** 0.5)


# Global handler instance
_GLOBAL_HANDLER = None
_ORIGINAL_CLIP_GRAD_NORM = None


def patch_mixed_precision(mixed_precision_target='auto'):
    """
    Patch PyTorch's gradient utilities to handle mixed precision automatically.

    Args:
        mixed_precision_target: 'auto', 'f32', 'bf16', 'f16'
            - 'auto': Smart conversion (f8→bf16, int8→f16)
            - 'f32': Convert all problematic dtypes to float32
            - 'bf16': Convert all problematic dtypes to bfloat16
            - 'f16': Convert all problematic dtypes to float16
    """
    global _GLOBAL_HANDLER, _ORIGINAL_CLIP_GRAD_NORM

    # Store original function BEFORE patching
    _ORIGINAL_CLIP_GRAD_NORM = torch.nn.utils.clip_grad_norm_

    # Create handler
    _GLOBAL_HANDLER = MixedPrecisionHandler(mixed_precision_target=mixed_precision_target)

    # Patch gradient clipping
    torch.nn.utils.clip_grad_norm_ = _GLOBAL_HANDLER.create_safe_clip_function(_ORIGINAL_CLIP_GRAD_NORM)

    print(f" Mixed precision gradient handling patched")
    print(f"   Target dtype strategy: {mixed_precision_target}")
    if mixed_precision_target == 'auto':
        print(f"   Auto conversions: f8→bf16, int8→f16")
    else:
        print(f"   All problematic dtypes → {mixed_precision_target}")

    return _GLOBAL_HANDLER


def get_mixed_precision_clip_function():
    """Returns the safe mixed device clip function for direct use"""
    return lambda params, max_norm, mpu=None: safe_mixed_device_clip_grad_norm(params, max_norm, _GLOBAL_HANDLER)


def unpatch_mixed_precision():
    """Restore original PyTorch functions"""
    global _GLOBAL_HANDLER, _ORIGINAL_CLIP_GRAD_NORM
    if _ORIGINAL_CLIP_GRAD_NORM:
        torch.nn.utils.clip_grad_norm_ = _ORIGINAL_CLIP_GRAD_NORM
        _GLOBAL_HANDLER = None
        print(" Mixed precision patches removed")


# Context manager for temporary patching
class MixedPrecisionContext:
    """
    Context manager for temporary mixed precision handling.

    Usage:
        with MixedPrecisionContext():
            # Mixed precision operations here
            pass
    """

    def __init__(self, mixed_precision_target='auto'):
        self.mixed_precision_target = mixed_precision_target
        self.original_handler = None

    def __enter__(self):
        self.original_handler = patch_mixed_precision(self.mixed_precision_target)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unpatch_mixed_precision()


if __name__ == "__main__":
    # Test the functionality
    print("Testing Mixed Precision Handler...")

    # Create test tensors with different dtypes
    test_param_f32 = torch.nn.Parameter(torch.randn(10, 10))
    test_param_f32.grad = torch.randn(10, 10)

    # Simulate float8 parameter (would normally cause error)
    test_param_f8 = torch.nn.Parameter(torch.randn(10, 10).to(torch.bfloat16))
    test_param_f8.grad = torch.randn(10, 10).to(torch.float8_e4m3fn) if hasattr(torch,
                                                                                'float8_e4m3fn') else torch.randn(10,
                                                                                                                  10).to(
        torch.bfloat16)

    # Test with different targets
    for target in ['auto', 'f32', 'bf16', 'f16']:
        print(f"\n--- Testing target: {target} ---")

        # Patch with target
        patch_mixed_precision(mixed_precision_target=target)

        # Test safe clipping
        try:
            norm = torch.nn.utils.clip_grad_norm_([test_param_f32, test_param_f8], 1.0)
            print(f" Safe clipping test passed, norm: {norm:.4f}")
        except Exception as e:
            print(f" Safe clipping test failed: {e}")

        # Cleanup
        unpatch_mixed_precision()

    print("Mixed Precision Handler test complete!")
