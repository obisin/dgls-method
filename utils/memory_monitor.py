from datetime import datetime, timezone, timedelta
import datetime
import subprocess
import torch
import gc
import psutil
import bitsandbytes as bnb

# =============================================================================
# PRECISION FRAMEWORK CODE
# =============================================================================

class MemoryMonitor:
    def __init__(self, expected_steps, transfer_stats, model_engine, add_smart_swapping_to_layer,  stats_window_size=10):

        self.expected_steps = expected_steps
        self.transfer_stats = transfer_stats  # Shared reference from main
        self.stats_window_size = stats_window_size
        self.model_engine = model_engine
        self.add_smart_swapping_to_layer = add_smart_swapping_to_layer

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
                'nf4': 'nf4',  # Special handling needed
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

                    # Also skip if this is wrapped by LoRA (check parent names)
                    parent_names = name.split('.')
                    if any(any(lora_term in part.lower() for lora_term in ['lora', 'adapter'])
                           for part in parent_names):
                        # print(f"   Skipping LoRA-wrapped layer: {name}")
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
            print(f" 4-bit replacement failed: {e}")
            return layer


def check_wsl_free_memory():
    """Quick WSL memory check using free command"""
    try:
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        print("\n WSL Memory (free -h):")
        print(result.stdout)

        # Get percentage
        result_percent = subprocess.run(['free'], capture_output=True, text=True)
        lines = result_percent.stdout.strip().split('\n')
        if len(lines) >= 2:
            values = lines[1].split()
            total, used = int(values[1]), int(values[2])
            percent = (used / total) * 100
            print(f"   Memory usage: {percent:.1f}%")
            return percent
    except Exception as e:
        print(f"Error checking memory: {e}")
        return None


def print_detailed_memory_stats(step_info=""):
    """Monitor both GPU and CPU memory"""
    print(f"\n Memory Stats {step_info}")

    # GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"   GPU Allocated: {allocated:.2f}GB")
        print(f"   GPU Reserved:  {reserved:.2f}GB")

    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"   System RAM:    {ram.used / 1e9:.1f}GB / {ram.total / 1e9:.1f}GB ({ram.percent:.1f}%)")
    print(f"   Swap Usage:    {swap.used / 1e9:.1f}GB / {swap.total / 1e9:.1f}GB ({swap.percent:.1f}%)")

    # Check if we're hitting swap (bad!)
    if swap.percent > 6.5:
        print(f"  WARNING: High swap usage ({swap.percent:.1f}%) - CPU memory pressure!")


def print_gpu_layer_state(stage, layer_idx, cpu_swappable_layers, layers, get_layer_device):
    """Print which swappable layers are currently on GPU"""
    gpu_layers = []
    for idx in sorted(cpu_swappable_layers):
        if idx < len(layers):
            device = get_layer_device(layers[idx])
            if device.type == 'cuda':
                gpu_layers.append(idx)

    print(f" GPU LAYERS {stage} layer {layer_idx}: {gpu_layers} (count: {len(gpu_layers)})")
    return gpu_layers

def aggressive_cpu_cleanup():
    """Aggressively clean up CPU memory to prevent swap pressure"""
    # Check if we're in trouble
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()

    if ram.percent > 90 or swap.percent > 30:
        # Force garbage collection
        gc.collect()

        # Clear PyTorch's CPU tensor cache if it exists
        if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
            try:
                torch._C._cuda_clearCublasWorkspaces()
            except:
                pass

        print(f" CPU cleanup: RAM {ram.percent:.1f}%, Swap {swap.percent:.1f}%")


