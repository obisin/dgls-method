import argparse
import os
import wandb
import datetime
import shutil
import glob
import json
import inspect
from pathlib import Path
from functools import wraps
import toml
import deepspeed
from deepspeed import comm as dist
from deepspeed.runtime.pipe import module as ds_pipe_module
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time
from utils import common
from utils.common import is_main_process, get_rank, DTYPE_MAP, empty_cuda_cache
import utils.saver
from utils.patches import apply_patches
from utils.unsloth_utils import unsloth_checkpoint
from utils.pipeline import ManualPipelineModule
from collections import defaultdict
import torch
import gc
import threading
import warnings
import psutil
import types
import bitsandbytes as bnb
import torch.profiler
import torch.cuda.nvtx as nvtx
import torch._dynamo
from functools import partial
import torch.utils.checkpoint
import utils.dataset as dataset_util
from utils.memory_monitor import check_wsl_free_memory, print_detailed_memory_stats, print_gpu_layer_state, aggressive_cpu_cleanup

warnings.filterwarnings("ignore", message="Padding mask is disabled")

from utils.mixed_precision_handler import (
    patch_mixed_precision,
    get_mixed_precision_clip_function,
    CastingHandler)

torch.utils.checkpoint.set_checkpoint_early_stop(False)

#Patch the checkpoint function to disable determinism checking # This disables the metadata checking
torch.utils.checkpoint.checkpoint = partial(
    torch.utils.checkpoint.checkpoint,
    determinism_check="none")

torch._dynamo.config.suppress_errors = True
wandb_enable = False

"""
Dynamic GPU Layer Swapping Trainer
==================================
Enhanced training with automatic memory optimization.

PERFORMANCE DESIGN NOTE:
This file uses module-level functions with local variable binding for optimal performance.
Hot path functions bind frequently-accessed attributes to locals at function start (LOAD_FAST)
to avoid Python's attribute lookup overhead (LOAD_ATTR). Call chains are minimized to 
reduce function call overhead.

Architecture:
- Globals: Intentionally used for performance (documented below)
- Hot functions: Bind globals to locals once per call
- Minimal call depth: Direct function calls without abstraction layers
- Factory pattern: Clean call sites with fast execution

CRITICAL PATH: Functions marked "HOT PATH" - profile before changing!
"""


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# PROCEED WITH CAUTION WITH ZERO OPTIMIZERS, SHARDED, GRADIENT RELEASE OR GENERICOPTIM!
# THESE WILL NOT PLAY WELL WITH LAYER SWAPPING.

TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

parser = argparse.ArgumentParser()

parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', nargs='?', const=True, default=None,
                    help='resume training from checkpoint. If no value is provided, resume from the most recent checkpoint. If a folder name is provided, resume from that specific folder.')
parser.add_argument('--regenerate_cache', action='store_true', help='Force regenerate cache.')
parser.add_argument('--cache_only', action='store_true', help='Cache model inputs then exit.')
parser.add_argument('--trust_cache', action='store_true',
                    help='Load from metadata cache files if they exist, without checking if any fingerprints have changed. Can make loading much faster for large datasets.')
parser.add_argument('--i_know_what_i_am_doing', action='store_true',
                    help="Skip certain checks and overrides. You may end up using settings that won't work.")
parser.add_argument('--master_port', type=int, default=29500, help='Master port for distributed training')
parser.add_argument('--dump_dataset', type=Path, default=None,
                    help='Decode cached latents and dump the dataset to this directory.')

parser.add_argument('--dynamic_swapping', action='store_true', default=True,
                    help='Smart dynamic layer swapping between GPU and CPU for optimal performance.')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (0, 1, 2, etc.)')
parser.add_argument('--device_sync_mode', choices=['early', 'late', 'off'], default='late',
                    help='When to sync devices: early (post-checkpoint: unstable on 2060), late (pre-optimizer,default, early causes instability in some GPUs), off (disabled)')
parser.add_argument('--sync_only_on_resume', action='store_true',
                    help='Only run device sync when resuming from checkpoint, not on fresh training')

parser.add_argument('--initial_gpu_layers', type=int, default=1,
                    help='Number of initial layers to keep permanently on GPU. '
                         'If not specified, uses reasonable defaults based on estimated VRAM.')
parser.add_argument('--final_gpu_layers', type=int, default=1,
                    help='Number of final layers to keep permanently on GPU.')
parser.add_argument('--gpu_layers', type=str, default=None,
                    help='Comma-separated list of layer indices to keep permanently on GPU '
                         '(e.g., "0,1,2,14,18,19,20,21,22"). Overrides initial_gpu_layers and final_gpu_layers '
                         'for precise control over which specific layers stay on GPU vs get swapped to CPU. '
                         'Use this to target problematic layer types while maximizing swappable layers. Your layer types will print at startup.')

parser.add_argument('--prefetch', type=int, nargs='?', const=1, default=0,
                    help='Number of layers to prefetch ahead (0=off, 1=single layer, 2+=multiple layers), might not work with mixed lyer type models')
parser.add_argument('--threading', action='store_true', default=False,
                   help='Enable background threading for automatic layer management')
parser.add_argument('--cuda_streams', action='store_true', default=False,
                   help='Enable CUDA streams for copy-compute overlap (needs more VRAM)')
parser.add_argument('--batch_move', action='store_true', default=False,
                   help='Use batch layer moving (experimental, may cause device issues)')

parser.add_argument('--cast_target', nargs=2, metavar=('FROM', 'TO'),
                   help='Cast FROM dtype TO dtype at start-up (e.g., f32 bf16) choices=[f32, bf16, f16, f8_e4m3, f8_e5m2, nf4, fp4]')

parser.add_argument('--selective_packing', type=int,nargs='?', const=64, default=0,
                   help='Size threshold in MB for using packed transfers (default: 64MB)')

parser.add_argument('--event_sync', action='store_true', default=False,
                   help='Use CUDA events instead of torch.cuda.synchronize() for better performance')

parser.add_argument('--async_zero_grad', action='store_true', default=False,
                   help='Zero gradients asynchronously on separate CUDA stream')

parser.add_argument('--lazy_lora_steps', type=int, default=1,
                   help='Update LoRA adapters every N steps instead of every step (1=disabled)')

parser.add_argument('--compile', action='store_true', default=False,
                   help='Use torch.compile optimization for resident GPU layers')

parser.add_argument('--autocast', choices=['fp32', 'fp16', 'bf16', 'f8_e4m3', 'f8_e5m2'], default=None,
                   help='Autocast precision for mixed precision training (fp32=disabled, fp16, bf16, f8_e4m3, f8_e5m2)')

parser.add_argument('--mixed_precision', choices=['auto', 'f32', 'bf16', 'f16'], default='auto',
                   help='Target dtype for problematic mixed precision conversions. '
                        'auto: smart fallback (f8→bf16, int8→f16), '
                        'f32/bf16/f16: convert all problematic dtypes to specified target. '
                        'Choices: auto, f32, bf16, f16')

parser.add_argument('--verbose', action='store_true', default=False,
                   help='Enable verbose output with detailed timing and transfer information')



# install_math_guard(mode='move') #set to move for production
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
patch_mixed_precision(mixed_precision_target=args.mixed_precision)

PROFILE = False
VERBOSE = args.verbose #False
TRACING_PROFILER = False

casting_handler = None
if args.cast_target:
    casting_handler = CastingHandler()

# =============================================================================
# GLOBALS MANIFEST (intentionally module-level for performance)
# =============================================================================

# Device configuration
PRIMARY_GPU_ID = 0
GPU_DEVICE = None  # Set after device detection
CPU_DEVICE = 'cpu'

# Swapping state
swap_stats = {'to_gpu': 0, 'to_cpu': 0}
transfer_events = {}
layer_sizes_mb = {}
expected_steps = 0

# Training state
ds_config = None
model_highest_precision = None
STATS_WINDOW_SIZE = 10

# Required for DeepSpeed to avoid "cannot load mpi library" error
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous CUDA

# Add global transfer and step stats
transfer_stats = {
    'to_gpu_times': [],
    'to_cpu_times': [],
    'to_gpu_speeds': [],
    'to_cpu_speeds': [],
    'current_step_gpu_times': [],  # Add these new fields
    'current_step_cpu_times': [],
    'current_step_gpu_speeds': [],
    'current_step_cpu_speeds': [],
    'step_durations': [],
    'step_total_transfer_times': [],
    'step_avg_transfer_speeds': []
}

gpu_resident_layers = set()
cpu_swappable_layers = set()
packed_layers = {}

device_cache = None
layers = None
model_engine = None

# Override with command line args if provided
if args.initial_gpu_layers is not None:
    initial_gpu_layers = args.initial_gpu_layers
    print(f" Overriding initial_gpu_layers: {initial_gpu_layers}")

if args.final_gpu_layers is not None:
    final_gpu_layers = args.final_gpu_layers
    print(f" Overriding final_gpu_layers: {final_gpu_layers}")
elif args.initial_gpu_layers is not None:
    # If only initial is specified, match final to initial
    final_gpu_layers = args.initial_gpu_layers
    print(f" Setting final_gpu_layers to match initial: {final_gpu_layers}")

print(" Dynamic GPU Layer Swapping Trainer")
print(f"   Initial GPU layers: {initial_gpu_layers}")
print(f"   Final GPU layers: {final_gpu_layers}")
print(f"   Prefetch: {args.prefetch}")
print(f"   Threading: {'enabled' if args.threading else 'disabled'}")

if torch.cuda.is_available():
    available_gpus = torch.cuda.device_count()
    print(f" Available GPUs: {available_gpus}")

    # Override with command line argument if provided
    if hasattr(args, 'gpu_id') and args.gpu_id != 0:  # Only override if explicitly set
        PRIMARY_GPU_ID = args.gpu_id
        print(f" Command line override: using GPU {PRIMARY_GPU_ID}")
    else:
        print(f" Using default GPU {PRIMARY_GPU_ID}")

    if PRIMARY_GPU_ID >= available_gpus:
        print(f"  GPU {PRIMARY_GPU_ID} not available, falling back to GPU 0")
        PRIMARY_GPU_ID = 0

    PRIMARY_GPU_DEVICE = f'cuda:{PRIMARY_GPU_ID}'
    print(f" Using GPU {PRIMARY_GPU_ID} ({PRIMARY_GPU_DEVICE})")

    # Set the primary device for CUDA operations
    torch.cuda.set_device(PRIMARY_GPU_ID)
else:
    PRIMARY_GPU_DEVICE = 'cpu'
    print(" No CUDA available, using CPU")

# Device strings for easy replacement
GPU_DEVICE = PRIMARY_GPU_DEVICE
CPU_DEVICE = 'cpu'

print(f" Device configuration: GPU={GPU_DEVICE}, CPU={CPU_DEVICE}")

import sys
def in_recompute(max_depth=12):
    f = sys._getframe(1)  # caller
    for _ in range(max_depth):
        name = f.f_code.co_name.lower()
        file = (f.f_code.co_filename or "").lower()
        if ('recompute' in name or 'backward' in name or 'autograd' in file):
            return True
        f = f.f_back
        if f is None:
            break
    return False

def get_cached_layer_size_mb(idx):
    return layer_sizes_mb.get(idx, 0)

def lazy_lora_step(model_engine, step):
    """Only run optimizer.step() every N steps for LoRA"""
    if step % args.lazy_lora_steps == 0:
        # Normal optimizer step
        if VERBOSE:
            start_time = time.time()
        model_engine.optimizer.step()
        if VERBOSE:
            optimizer_time = time.time() - start_time

        if not hasattr(model_engine, 'optimizer_times'):
            model_engine.optimizer_times = []
        if VERBOSE:
            model_engine.optimizer_times.append(optimizer_time)
            print(f" LoRA optimizer step: {optimizer_time:.4f}s")
        return True
    else:
        # Skip optimizer step, but still zero gradients
        model_engine.optimizer.zero_grad()
        if VERBOSE:
            print(f" Skipped LoRA optimizer step {step}")

        # Add zero time to keep timing consistent
        if not hasattr(model_engine, 'optimizer_times'):
            model_engine.optimizer_times = []
        if VERBOSE:
            model_engine.optimizer_times.append(0.0)

        return False

def async_zero_gradients(model_engine):
    """Zero gradients asynchronously if enabled"""
    if (args.async_zero_grad and args.cuda_streams and
            hasattr(add_smart_swapping_to_layer, 'async_zero_stream')):
        with torch.cuda.stream(add_smart_swapping_to_layer.async_zero_stream):
            model_engine.optimizer.zero_grad()
        zero_event = torch.cuda.Event()
        zero_event.record(add_smart_swapping_to_layer.async_zero_stream)
        torch.cuda.current_stream().wait_event(zero_event)
    else:
        model_engine.optimizer.zero_grad()

def reset_current_step_stats():
    """Reset stats for the current step"""
    transfer_stats['current_step_gpu_times'] = []
    transfer_stats['current_step_cpu_times'] = []
    transfer_stats['current_step_gpu_speeds'] = []
    transfer_stats['current_step_cpu_speeds'] = []


def calculate_step_transfer_stats():
    """Calculate transfer stats for the current step"""
    all_times = transfer_stats['current_step_gpu_times'] + transfer_stats['current_step_cpu_times']
    all_speeds = transfer_stats['current_step_gpu_speeds'] + transfer_stats['current_step_cpu_speeds']

    if not all_times:
        return None, None, None

    total_transfer_time = sum(all_times)
    avg_transfer_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0
    avg_transfer_duration = sum(all_times) / len(all_times) if all_times else 0

    return total_transfer_time, avg_transfer_speed, avg_transfer_duration


def track_step_performance(step_duration, step_num, layer_compute_time, end_of_training):
    """Track step duration with complete timing breakdown"""
    # Calculate current step transfer stats
    total_transfer_time, avg_transfer_speed, avg_transfer_duration = calculate_step_transfer_stats()

    transfer_stats['step_durations'].append(step_duration)

    if total_transfer_time is not None:
        transfer_stats['step_total_transfer_times'].append(total_transfer_time)
        transfer_stats['step_avg_transfer_speeds'].append(avg_transfer_speed)

    if step_num % STATS_WINDOW_SIZE == 0 or end_of_training:
        # Get available data, up to STATS_WINDOW_SIZE
        window_size = min(STATS_WINDOW_SIZE, len(transfer_stats['step_durations']))
        recent_step_times = transfer_stats['step_durations'][-window_size:]
        avg_step_time = sum(recent_step_times) / len(recent_step_times)

        if transfer_stats['step_avg_transfer_speeds']:
            recent_transfer_speeds = transfer_stats['step_avg_transfer_speeds'][-window_size:]
            avg_transfer_speed_windowed = sum(recent_transfer_speeds) / len(recent_transfer_speeds)

            recent_transfer_times = transfer_stats['step_total_transfer_times'][-window_size:]
            avg_transfer_time_windowed = sum(recent_transfer_times) / len(recent_transfer_times)

            total_seconds = avg_step_time * expected_steps
            remaining_seconds = (expected_steps - step_num) * avg_step_time
            expected_total_str = str(datetime.timedelta(seconds=int(total_seconds)))
            time_remaining_str = str(datetime.timedelta(seconds=int(remaining_seconds)))

            print(
                f" Last {window_size} steps avg: {avg_step_time:.1f}s step duration, {avg_transfer_time_windowed:.2f}s transfer time, {avg_transfer_speed_windowed:.0f} MB/s, "
                f"Expected total time: {expected_total_str}, "
                f"Remaining Steps: {expected_steps - step_num}, "
                f"Time remaining: {time_remaining_str}")

    # Reset for next step
    reset_current_step_stats()
    # Clear DeepSpeed times for next step
    if hasattr(model_engine, 'optimizer_times'):
        model_engine.optimizer_times = []
    if hasattr(model_engine, 'backward_times'):
        model_engine.backward_times = []
    add_smart_swapping_to_layer.layer_compute_times = []


def event_based_sync(operation_name, idx=None):
    """Replace torch.cuda.synchronize() with specific event tracking"""
    if args.event_sync:
        event = torch.cuda.Event()
        event.record()

        key = f"{operation_name}_{idx}" if idx is not None else operation_name
        transfer_events[key] = event
        return event
    else:
        torch.cuda.synchronize()
        return None


def wait_for_event(operation_name, idx=None):
    """Wait for specific event to complete"""
    if args.event_sync:
        key = f"{operation_name}_{idx}" if idx is not None else operation_name
        if key in transfer_events:
            transfer_events[key].wait()
            del transfer_events[key]


class LayerDeviceCache:
    def __init__(self, model, layers):  # Add layers parameter
        self.cache = {}
        self.dirty = set()
        # Initialize cache using the layers list
        for i, layer in enumerate(layers):
            self.cache[i] = self.get_layer_device(layer)

    def get_layer_device(self, layer):
        """Your existing function"""
        try:
            return next(layer.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def get_device(self, layer_idx):  # Simplified - no need for model parameter
        """Fast cached lookup"""
        if layer_idx in self.dirty:
            self.cache[layer_idx] = self.get_layer_device(layers[layer_idx])
            self.dirty.remove(layer_idx)
        return self.cache[layer_idx]

    def mark_moved(self, layer_idx, new_device):
        """Update cache when we move a layer"""
        self.cache[layer_idx] = new_device



def safe_move_to_cpu(layer, idx):
    """Move layer back to CPU and clean up GPU memory"""
    # Check if layer is already on CPU
    if PROFILE: nvtx.range_push(f"CPU_Transfer_L{idx}")
    try:
        try:
            current_device = next(layer.parameters()).device
            if current_device.type == 'cpu':
                return  # Already on CPU
        except StopIteration:
            # Layer has no parameters
            pass

        transfer_time = 0

        if VERBOSE:
            start_time = time.time()
            layer_size_mb = get_cached_layer_size_mb(idx)
        layer.to(CPU_DEVICE)
        add_smart_swapping_to_layer.cleanup_stream = torch.cuda.Stream()
        swap_stats['to_cpu'] += 1
        if VERBOSE:
            end_time = time.time()
            transfer_time = end_time - start_time

        if transfer_time > 0 and VERBOSE:
            speed_mbps = layer_size_mb / transfer_time

            # Track current step stats
            transfer_stats['current_step_cpu_times'].append(transfer_time)
            transfer_stats['current_step_cpu_speeds'].append(speed_mbps)

        # if idx % 20 == 0:
        #     torch.cuda.empty_cache()
        device_cache.mark_moved(idx, torch.device('cpu'))

        # Only log every 10th layer or if there are issues
        # if idx % 10 == 0:
        #     print(f"   Layer {idx} → CPU, memory: {torch.cuda.memory_allocated(0) / 1e9:.1f}GB")

        return True
    finally:
        if PROFILE: nvtx.range_pop()


def safe_move_to_gpu(layer, idx):
    """Move layer to GPU with dtype casting"""
    if PROFILE: nvtx.range_push(f"GPU_Transfer_L{idx}")
    try:
        try:

            current_device = device_cache.get_device(idx)
            if current_device.type == 'cuda':
                return True

            if VERBOSE:
                start_time = time.time()
                layer_size_mb = get_cached_layer_size_mb(idx)

            layer.to(GPU_DEVICE, non_blocking=True)

            if VERBOSE:
                end_time = time.time()
                transfer_time = end_time - start_time

            if transfer_time > 0 and VERBOSE:
                speed_mbps = layer_size_mb / transfer_time
                # Track current step stats
                transfer_stats['current_step_gpu_times'].append(transfer_time)
                transfer_stats['current_step_gpu_speeds'].append(speed_mbps)

            event_based_sync("gpu_transfer", idx)
            device_cache.mark_moved(idx, torch.device('cuda'))
            swap_stats['to_gpu'] += 1
            return True

        except RuntimeError as e:
            if "out of memory" in str(e):
                return False
            raise e
    finally:
        if PROFILE: nvtx.range_pop()


# THREADING: Thread-safe GPU operations
def safe_move_to_gpu_threaded(layer, idx):
    """Thread-safe move to GPU"""
    with add_smart_swapping_to_layer.gpu_lock:
        if device_cache.get_device(idx).type == 'cpu':
            return safe_move_to_gpu(layer, idx)
        return True


def safe_move_to_cpu_threaded(layer, idx):
    """Thread-safe move to CPU"""
    with add_smart_swapping_to_layer.gpu_lock:
        if device_cache.get_device(idx).type == 'cuda':
            return safe_move_to_cpu(layer, idx)
        return True

def stop_background_threading():
    """Stop the background threading system"""
    if hasattr(add_smart_swapping_to_layer, 'training_active'):
        add_smart_swapping_to_layer.training_active = False
        if add_smart_swapping_to_layer.background_thread:
            add_smart_swapping_to_layer.background_thread.join(timeout=1.0)
        print(" Background threading stopped")



def calculate_needed_layers(layer_idx, is_backward, prefetch):
    """Calculate which layers we need on GPU for current operation"""
    needed = set()
    needed.add(layer_idx)  # Always need current layer

    if is_backward:
        # Backward pass: need previous layer for activations
        if layer_idx - 1 in cpu_swappable_layers and layer_idx - 1 >= 0:
            needed.add(layer_idx - 1)

        # Add prefetch buffer (going backward)
        for i in range(1, prefetch + 1):
            prefetch_idx = layer_idx - i - 1
            if prefetch_idx in cpu_swappable_layers and prefetch_idx >= 0:
                needed.add(prefetch_idx)
    else:
        # Forward pass: add prefetch buffer (going forward)
        for i in range(1, prefetch + 1):
            prefetch_idx = layer_idx + i
            if prefetch_idx in cpu_swappable_layers and prefetch_idx < len(layers):
                needed.add(prefetch_idx)

    return needed


def cleanup_excess_layers(keep_layers):
    """Remove layers from GPU that are not in keep_layers set"""
    if PROFILE: nvtx.range_push(f"Cleanup_Excess_{len(cpu_swappable_layers - keep_layers)}layers")
    try:
        if args.batch_move:
            # Batch approach (your current code)
            layers_to_remove = []
            for idx in cpu_swappable_layers:
                if (idx < len(layers) and
                        idx not in keep_layers and
                        device_cache.get_device(idx).type == 'cuda'):
                    layers_to_remove.append(idx)
            cleaned_count = batch_safe_move_to_cpu(layers_to_remove)
        else:
            # Individual approach (fallback)
            cleaned_count = 0
            for idx in cpu_swappable_layers:
                if (idx < len(layers) and
                        idx not in keep_layers and
                        device_cache.get_device(idx).type == 'cuda'):
                    safe_move_to_cpu(layers[idx], idx)
                    cleaned_count += 1

        return cleaned_count
    finally:
        if PROFILE: nvtx.range_pop()


def fetch_missing_layers(needed_layers):
    """Ensure all needed layers are on GPU"""
    if PROFILE: nvtx.range_push(f"Fetch_Missing_{len(needed_layers)}layers")
    try:
        if args.batch_move:
            # Batch approach
            layers_to_fetch = []
            for idx in needed_layers:
                if (idx < len(layers) and
                        idx in cpu_swappable_layers and
                        device_cache.get_device(idx).type == 'cpu'):
                    layers_to_fetch.append(idx)
            fetched_count = batch_safe_move_to_gpu(layers_to_fetch)
        else:
            # Individual approach (more stable)
            fetched_count = 0
            for idx in needed_layers:
                if (idx < len(layers) and
                        idx in cpu_swappable_layers and
                        device_cache.get_device(idx).type == 'cpu'):
                    success = safe_move_to_gpu(layers[idx], idx)
                    if success:
                        fetched_count += 1

        return fetched_count
    finally:
        if PROFILE: nvtx.range_pop()


def batch_safe_move_to_gpu_packed(layer_indices, threshold_mb=64):
    """Batch move to GPU with selective packing"""
    if not layer_indices:
        return 0

    moved_count = 0
    large_layers = []
    small_layers = []

    # Separate layers by size
    for idx in layer_indices:
        if (idx < len(layers) and
                device_cache.get_device(idx).type == 'cpu'):

            layer_size_mb = get_cached_layer_size_mb(idx)

            if layer_size_mb > threshold_mb and idx in packed_layers:
                large_layers.append(idx)
            else:
                small_layers.append(idx)

    # Handle large layers with packed transfer
    if large_layers:
        for idx in large_layers:
            success = safe_move_to_gpu_packed(layers[idx], idx)
            if success:
                moved_count += 1

    # Handle small layers with direct transfer
    if small_layers:
        for idx in small_layers:
            success = safe_move_to_gpu(layers[idx], idx)
            if success:
                moved_count += 1

    return moved_count


def batch_safe_move_to_cpu_packed(layer_indices, threshold_mb=64):
    """Batch move to CPU with selective packing"""
    if not layer_indices:
        return 0

    moved_count = 0
    large_layers = []
    small_layers = []

    # Separate layers by size
    for idx in layer_indices:
        if (idx < len(layers) and
                device_cache.get_device(idx).type == 'cuda'):

            layer_size_mb = get_cached_layer_size_mb(idx)

            if layer_size_mb > threshold_mb:
                large_layers.append(idx)
            else:
                small_layers.append(idx)

    # Handle large layers with packed transfer
    if large_layers:
        for idx in large_layers:
            success = safe_move_to_cpu_packed(layers[idx], idx)
            if success:
                moved_count += 1

    # Handle small layers with direct transfer
    if small_layers:
        for idx in small_layers:
            success = safe_move_to_cpu(layers[idx], idx)
            if success:
                moved_count += 1

    return moved_count


def cleanup_excess_layers_packed(keep_layers, threshold_mb=64):
    """Remove layers from GPU with selective packing awareness"""

    if args.batch_move:
        # Collect all layers to remove, then use batch function
        layers_to_remove = []
        for idx in cpu_swappable_layers:
            if (idx < len(layers) and
                    idx not in keep_layers and
                    device_cache.get_device(idx).type == 'cuda'):
                layers_to_remove.append(idx)
        return batch_safe_move_to_cpu_packed(layers_to_remove, threshold_mb)

    else:
        # Individual processing
        moved_count = 0
        for idx in cpu_swappable_layers:
            if (idx < len(layers) and
                    idx not in keep_layers and
                    device_cache.get_device(idx).type == 'cuda'):

                layer_size_mb = get_cached_layer_size_mb(idx)

                if layer_size_mb > threshold_mb:
                    success = safe_move_to_cpu_packed(layers[idx], idx)
                else:
                    success = safe_move_to_cpu(layers[idx], idx)

                if success:
                    moved_count += 1

        return moved_count


def fetch_missing_layers_packed(needed_layers, threshold_mb=64):
    """Use packed transfers only for large layers, direct for small ones"""

    if args.batch_move:
        # Use batch function
        return batch_safe_move_to_gpu_packed(needed_layers, threshold_mb)

    else:
        # Individual processing
        fetched_count = 0
        for idx in needed_layers:
            if (idx < len(layers) and
                    idx in cpu_swappable_layers and
                    device_cache.get_device(idx).type == 'cpu'):

                layer_size_mb = get_cached_layer_size_mb(idx)

                if layer_size_mb > threshold_mb and idx in packed_layers:
                    success = safe_move_to_gpu_packed(layers[idx], idx)
                else:
                    success = safe_move_to_gpu(layers[idx], idx)

                if success:
                    fetched_count += 1

        return fetched_count


def background_layer_manager():
    """Background thread that maintains sliding window of layers"""
    print("🧵 Background layer manager started")

    while add_smart_swapping_to_layer.training_active:
        try:
            if PROFILE: nvtx.range_push("Background_Manager_Cycle")
            current_idx = add_smart_swapping_to_layer.current_layer_idx
            is_backward = add_smart_swapping_to_layer.is_backward_pass

            needed_layers = calculate_needed_layers(current_idx, is_backward, args.prefetch)

            # Collect layers to move instead of moving one by one
            layers_to_cpu = []
            layers_to_gpu = []

            if args.batch_move:
                # One in, one out: remove old layers, add new ones
                for idx in cpu_swappable_layers:
                    if idx not in needed_layers and device_cache.get_device(idx).type == 'cuda':
                        layers_to_cpu.append(idx)

                for idx in needed_layers:
                    if device_cache.get_device(idx).type == 'cpu':
                        layers_to_gpu.append(idx)

                # Batch move with thread safety
                if layers_to_cpu or layers_to_gpu:
                    with add_smart_swapping_to_layer.gpu_lock:
                        if layers_to_cpu:
                            batch_safe_move_to_cpu(layers_to_cpu)
                        if layers_to_gpu:
                            batch_safe_move_to_gpu(layers_to_gpu)
            else:
                # One in, one out: remove old layers, add new ones
                for idx in cpu_swappable_layers:
                    if idx not in needed_layers and device_cache.get_device(idx).type == 'cuda':
                        safe_move_to_cpu_threaded(layers[idx], idx)

                for idx in needed_layers:
                    if device_cache.get_device(idx).type == 'cpu':
                        safe_move_to_gpu_threaded(layers[idx], idx)

            time.sleep(0.000001) #do not touch. very needed for sync

        except Exception as e:
            print(f" Background thread error: {e}")
            time.sleep(0.1)
        finally:
            if PROFILE: nvtx.range_pop()


def fast_layer_transfer(layer, target_device):
    """Direct parameter movement without .to() overhead"""
    for param in layer.parameters():
        if param.device != target_device:
            param.data = param.data.to(target_device, non_blocking=True)

    for buffer in layer.buffers():
        if buffer.device != target_device:
            buffer.data = buffer.data.to(target_device, non_blocking=True)


def batch_safe_move_to_cpu(layer_indices):
    """Move multiple layers to CPU in batch"""
    if not layer_indices:
        return 0
    if PROFILE: nvtx.range_push(f"Batch_CPU_Transfer_{len(layer_indices)}layers")
    try:
        moved_count = 0
        for idx in layer_indices:
            if (idx < len(layers) and
                    device_cache.get_device(idx).type == 'cuda'):
                # Use your existing fast_layer_transfer or layer.to()
                # Measure transfer timing
                transfer_time = 0
                if VERBOSE:
                    start_time = time.time()
                    layer_size_mb = get_cached_layer_size_mb(idx)
                layers[idx].to(CPU_DEVICE)
                if VERBOSE:
                    end_time = time.time()
                    transfer_time = end_time - start_time

                if transfer_time > 0 and VERBOSE:
                    speed_mbps = layer_size_mb / transfer_time

                    # Track current step stats
                    transfer_stats['current_step_cpu_times'].append(transfer_time)
                    transfer_stats['current_step_cpu_speeds'].append(speed_mbps)
                device_cache.mark_moved(idx, torch.device('cpu'))
                swap_stats['to_cpu'] += 1
                moved_count += 1

        return moved_count
    finally:
        if PROFILE: nvtx.range_pop()

def batch_safe_move_to_gpu(layer_indices):
    """Batch move to GPU with dtype casting"""
    if not layer_indices:
        return 0

    if PROFILE: nvtx.range_push(f"Batch_GPU_Transfer_{len(layer_indices)}layers")
    try:

        moved_count = 0
        for idx in layer_indices:
            if (idx < len(layers) and
                    device_cache.get_device(idx).type == 'cpu'):

                layer = layers[idx]
                transfer_time = 0

                if VERBOSE:
                    start_time = time.time()
                    layer_size_mb = get_cached_layer_size_mb(idx)

                layer.to(GPU_DEVICE, non_blocking=True)


                if VERBOSE:
                    end_time = time.time()
                    transfer_time = end_time - start_time

                event_based_sync("gpu_transfer", idx)
                if transfer_time > 0 and VERBOSE:
                    speed_mbps = layer_size_mb / transfer_time
                    # Track current step stats
                    transfer_stats['current_step_gpu_times'].append(transfer_time)
                    transfer_stats['current_step_gpu_speeds'].append(speed_mbps)

                device_cache.mark_moved(idx, torch.device('cuda'))
                swap_stats['to_gpu'] += 1
                moved_count += 1

        return moved_count
    finally:
        if PROFILE: nvtx.range_pop()

# =============================================================================
# PACKED CODE
# =============================================================================

class PackedCPUBlock:
    def __init__(self, layer):
        self.packed_buffers = {}  # One buffer per dtype
        self.tensor_specs = []
        self.total_elements = 0
        self.gpu_blocks = {}  # Keep GPU buffers alive when resident
        self.gpu_events = {}  # CUDA events for synchronization
        self.pack_layer(layer)

    def pack_layer(self, layer):
        """Pack all layer parameters/buffers into contiguous CPU blocks by dtype"""
        dtype_groups = {}

        # Build param/buffer maps once (avoid rebuilding in loops)
        param_map = layer._parameters
        buffer_map = layer._buffers

        # Collect parameters by dtype
        for name, param in layer.named_parameters(recurse=False):
            if param is None:
                continue

            # Move to CPU first if needed (read-only copy during collection)
            src_data = param.detach().to('cpu', copy=True) if param.device.type != 'cpu' else param.data
            dtype = param.dtype

            if dtype not in dtype_groups:
                dtype_groups[dtype] = []
            dtype_groups[dtype].append({
                'name': name,
                'data': src_data.flatten(),
                'shape': param.shape,
                'is_param': True,
                'param_ref': param
            })

        # Collect buffers by dtype
        for name, buffer in layer.named_buffers(recurse=True):
            if buffer is None:
                continue

            # Move to CPU first if needed (read-only copy during collection)
            src_data = buffer.detach().to('cpu', copy=True) if buffer.device.type != 'cpu' else buffer.data
            dtype = buffer.dtype

            if dtype not in dtype_groups:
                dtype_groups[dtype] = []
            dtype_groups[dtype].append({
                'name': name,
                'data': src_data.flatten(),
                'shape': buffer.shape,
                'is_param': False,
                'buffer_ref': buffer
            })

        # Pack each dtype group into contiguous buffers
        for dtype, tensors in dtype_groups.items():
            if not tensors:
                continue

            total_size = sum(t['data'].numel() for t in tensors)

            packed_buffer = torch.empty(total_size, dtype=dtype)

            self.packed_buffers[dtype] = packed_buffer

            # Pack data and create specs
            offset = 0
            for tensor_info in tensors:
                data = tensor_info['data']
                size = data.numel()

                # Copy into packed buffer
                packed_buffer[offset:offset + size] = data

                # Store spec for later rebinding
                spec = {
                    'name': tensor_info['name'],
                    'dtype': dtype,
                    'offset': offset,
                    'size': size,
                    'shape': tensor_info['shape'],
                    'is_param': tensor_info['is_param'],
                    'param_ref': tensor_info.get('param_ref'),
                    'buffer_ref': tensor_info.get('buffer_ref')
                }
                self.tensor_specs.append(spec)
                offset += size

            self.total_elements += total_size

        # ATOMIC COMMIT: Rebind CPU params/buffers to views into packed_buffers
        # This eliminates duplicate host RAM
        for spec in self.tensor_specs:
            dtype = spec['dtype']
            start_idx = spec['offset']
            end_idx = start_idx + spec['size']
            cpu_view = self.packed_buffers[dtype][start_idx:end_idx].view(spec['shape'])

            if spec['is_param']:
                spec['param_ref'].data = cpu_view
            else:
                spec['buffer_ref'].data = cpu_view

    def unpack_to_gpu(self, layer):
        """Move packed buffers to GPU and rebind layer parameters"""
        # Group specs by dtype for efficient transfer
        dtype_specs = {}
        for spec in self.tensor_specs:
            dtype = spec['dtype']
            if dtype not in dtype_specs:
                dtype_specs[dtype] = []
            dtype_specs[dtype].append(spec)

        # Transfer each dtype group
        for dtype, specs in dtype_specs.items():
            if dtype not in self.packed_buffers:
                continue

            packed_cpu = self.packed_buffers[dtype]

            gpu_buffer = pinned_staging.to(GPU_DEVICE, non_blocking=True)

            # Record event and keep buffer alive
            event = torch.cuda.Event()
            event.record()
            self.gpu_blocks[dtype] = gpu_buffer
            self.gpu_events[dtype] = event

            # ATOMIC COMMIT: Rebind all params/buffers for this dtype
            for spec in specs:
                start_idx = spec['offset']
                end_idx = start_idx + spec['size']
                gpu_view = gpu_buffer[start_idx:end_idx].view(spec['shape'])

                if spec['is_param']:
                    spec['param_ref'].data = gpu_view
                else:
                    spec['buffer_ref'].data = gpu_view

    def unpack_to_cpu(self, layer):
        """Move GPU data back to packed CPU buffers and rebind layer parameters"""
        # Group specs by dtype
        dtype_specs = {}
        for spec in self.tensor_specs:
            dtype = spec['dtype']
            if dtype not in dtype_specs:
                dtype_specs[dtype] = []
            dtype_specs[dtype].append(spec)

        # Copy GPU → CPU for each dtype
        for dtype, specs in dtype_specs.items():
            if dtype not in self.gpu_blocks:
                continue

            gpu_buffer = self.gpu_blocks[dtype]
            packed_cpu = self.packed_buffers[dtype]

            # Synchronous copy back to packed CPU buffer
            packed_cpu.copy_(gpu_buffer.to('cpu'))
            # packed_cpu.copy_(gpu_buffer)

            # ATOMIC COMMIT: Rebind params/buffers back to CPU views
            for spec in specs:
                start_idx = spec['offset']
                end_idx = start_idx + spec['size']
                cpu_view = packed_cpu[start_idx:end_idx].view(spec['shape'])

                if spec['is_param']:
                    spec['param_ref'].data = cpu_view
                else:
                    spec['buffer_ref'].data = cpu_view

        # Release GPU memory
        self.gpu_blocks.clear()
        self.gpu_events.clear()

    def wait_for_gpu_transfer(self):
        """Wait for all async GPU transfers to complete"""
        for event in self.gpu_events.values():
            event.wait()

    def get_memory_usage(self):
        """Return memory usage info"""
        cpu_bytes = sum(buf.numel() * buf.element_size() for buf in self.packed_buffers.values())
        gpu_bytes = sum(buf.numel() * buf.element_size() for buf in self.gpu_blocks.values())
        return {
            'cpu_bytes': cpu_bytes,
            'gpu_bytes': gpu_bytes,
            'total_elements': self.total_elements
        }


def safe_move_to_gpu_packed(layer, idx):
    """Move layer to GPU using pre-packed buffer"""
    if PROFILE: nvtx.range_push(f"GPU_Transfer_Packed_L{idx}")
    try:
        current_device = device_cache.get_device(idx)
        if current_device.type == 'cuda':
            return True

        if VERBOSE:
            start_time = time.time()
            layer_size_mb = get_cached_layer_size_mb(idx)

        # if idx in packed_layers:
        #     # Use packed transfer - single GPU copy
        #     packed_layers[idx].unpack_to_gpu(layer, GPU_DEVICE)
        # else:
        #     # Fallback to normal transfer
        #     layer.to(GPU_DEVICE, non_blocking=True)

        if idx in packed_layers:
            # Use packed transfer (automatically uses pinned if enabled)
            packed_layers[idx].unpack_to_gpu(layer, GPU_DEVICE)
        else:
            # Fallback to normal transfer
            layer.to(GPU_DEVICE, non_blocking=True)

        transfer_time = 0

        event_based_sync("gpu_transfer", idx)
        if VERBOSE:
            end_time = time.time()
            transfer_time = end_time - start_time
        if transfer_time > 0 and VERBOSE:
            speed_mbps = layer_size_mb / transfer_time

            # Track current step stats
            transfer_stats['current_step_gpu_times'].append(transfer_time)
            transfer_stats['current_step_gpu_speeds'].append(speed_mbps)
        device_cache.mark_moved(idx, torch.device('cuda'))
        swap_stats['to_gpu'] += 1
        return True

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f" OOM moving packed layer {idx}")
            return False
        raise e
    finally:
        if PROFILE: nvtx.range_pop()


def safe_move_to_cpu_packed(layer, idx):
    """Move layer to CPU - packed version already stored"""
    if PROFILE: nvtx.range_push(f"CPU_Transfer_Packed_L{idx}")
    try:
        try:
            current_device = next(layer.parameters()).device
            if current_device.type == 'cpu':
                return True
        except StopIteration:
            pass

        transfer_time = 0

        if VERBOSE:
            start_time = time.time()
            layer_size_mb = get_cached_layer_size_mb(idx)
        layer.to(CPU_DEVICE)
        if VERBOSE:
            end_time = time.time()
            transfer_time = end_time - start_time

        if transfer_time > 0 and VERBOSE:
            speed_mbps = layer_size_mb / transfer_time
            # Track current step stats
            transfer_stats['current_step_cpu_times'].append(transfer_time)
            transfer_stats['current_step_cpu_speeds'].append(speed_mbps)
        device_cache.mark_moved(idx, torch.device('cpu'))
        swap_stats['to_cpu'] += 1
        return True
    finally:
        if PROFILE: nvtx.range_pop()

# =============================================================================
# ORIGINAL TRAINING FRAMEWORK CODE
# =============================================================================

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.state = defaultdict(dict)
        self.param_groups = []

    def step(self, closure=None):
        pass

    def zero_grad(self, set_to_none: bool = True):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


# Monkeypatch this so it counts all layer parameters, not just trainable parameters.
# This helps it divide the layers between GPUs more evenly when training a LoRA.
def _count_all_layer_params(self):
    param_counts = [0] * len(self._layer_specs)
    for idx, layer in enumerate(self._layer_specs):
        if isinstance(layer, ds_pipe_module.LayerSpec):
            l = layer.build()
            param_counts[idx] = sum(p.numel() for p in l.parameters())
        elif isinstance(layer, nn.Module):
            param_counts[idx] = sum(p.numel() for p in layer.parameters())
    return param_counts


ds_pipe_module.PipelineModule._count_layer_params = _count_all_layer_params


def set_config_defaults(config):
    # Force the user to set this. If we made it a default of 1, it might use a lot of disk space.
    assert 'save_every_n_epochs' in config or 'save_every_n_steps' in config

    config.setdefault('pipeline_stages', 1)
    config.setdefault('activation_checkpointing', False)
    config.setdefault('reentrant_activation_checkpointing', False)
    if config['activation_checkpointing'] == 'unsloth':
        config['reentrant_activation_checkpointing'] = True
    config.setdefault('warmup_steps', 0)
    if 'save_dtype' in config:
        config['save_dtype'] = DTYPE_MAP[config['save_dtype']]

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := model_config.get('transformer_dtype', None):
        model_config['transformer_dtype'] = DTYPE_MAP.get(transformer_dtype, transformer_dtype)
    model_config.setdefault('guidance', 1.0)

    if 'adapter' in config:
        adapter_config = config['adapter']
        adapter_type = adapter_config['type']
        if adapter_config['type'] == 'lora':
            if 'alpha' in adapter_config:
                raise NotImplementedError(
                    'This script forces alpha=rank to make the saved LoRA format simpler and more predictable with downstream inference programs. Please remove alpha from the config.'
                )
            adapter_config['alpha'] = adapter_config['rank']
            adapter_config.setdefault('dropout', 0.0)
            adapter_config.setdefault('dtype', model_dtype_str)
            adapter_config['dtype'] = DTYPE_MAP[adapter_config['dtype']]
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    config.setdefault('logging_steps', 1)
    config.setdefault('eval_datasets', [])
    config.setdefault('eval_gradient_accumulation_steps', 1)
    config.setdefault('eval_every_n_steps', None)
    config.setdefault('eval_every_n_epochs', None)
    config.setdefault('eval_before_first_step', True)
    config.setdefault('compile', False)


def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]


def print_model_info(model):
    if not is_main_process():
        return
    print(model)
    for name, module in model.named_modules():
        print(f'{type(module)}: {name}')
        for pname, p in module.named_parameters(recurse=False):
            print(pname)
            print(p.dtype)
            print(p.device)
            print(p.requires_grad)
            print()


# Need to preload all micro batches since pulling from the dataloader does IPC between the
# first and last stage. Can't do that during the train or inference pipeline schedule execution
# because it conflicts with the send / recv steps.
def get_data_iterator_for_step(dataloader, engine, num_micro_batches=None):
    num_micro_batches = num_micro_batches or engine.micro_batches
    if not (engine.is_first_stage() or engine.is_last_stage()):
        return None
    dataloader_iter = iter(dataloader)
    items = [next(dataloader_iter) for _ in range(num_micro_batches)]
    return iter(items)


def distributed_init(args):
    """Initialize distributed training environment."""
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    local_rank = args.local_rank

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(args.master_port)

    return world_size, rank, local_rank


def get_prodigy_d(optimizer):
    d = 0
    for group in optimizer.param_groups:
        d += group['d']
    return d / len(optimizer.param_groups)


def _get_automagic_lrs(optimizer):
    lrs = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            lr = optimizer._get_lr(group, state)
            lrs.append(lr)
    lrs = torch.stack(lrs)
    return lrs, lrs.mean()


if __name__ == '__main__':
    apply_patches()

    with open(args.config) as f:
        # Inline TOML tables are not pickleable, which messes up the multiprocessing dataset stuff. This is a workaround.
        config = json.loads(json.dumps(toml.load(f)))


    set_config_defaults(config)
    common.AUTOCAST_DTYPE = config['model']['dtype']
    dataset_util.UNCOND_FRACTION = config.get('uncond_fraction', 0.0)
    if map_num_proc := config.get('map_num_proc', None):
        dataset_util.NUM_PROC = map_num_proc

    # Initialize distributed environment before deepspeed
    world_size, rank, local_rank = distributed_init(args)

    # Now initialize deepspeed
    deepspeed.init_distributed()

    # needed for broadcasting Queue in dataset.py
    torch.cuda.set_device(dist.get_rank())

    resume_from_checkpoint = (
        args.resume_from_checkpoint if args.resume_from_checkpoint is not None
        else config.get('resume_from_checkpoint', False)
    )
    regenerate_cache = (
        args.regenerate_cache if args.regenerate_cache is not None
        else config.get('regenerate_cache', False)
    )

    model_type = config['model']['type']

    if model_type == 'flux':
        from models import flux

        model = flux.FluxPipeline(config)
    elif model_type == 'ltx-video':
        from models import ltx_video

        model = ltx_video.LTXVideoPipeline(config)
    elif model_type == 'hunyuan-video':
        from models import hunyuan_video

        model = hunyuan_video.HunyuanVideoPipeline(config)
    elif model_type == 'sdxl':
        from models import sdxl

        model = sdxl.SDXLPipeline(config)
    elif model_type == 'cosmos':
        from models import cosmos

        model = cosmos.CosmosPipeline(config)
    elif model_type == 'lumina_2':
        from models import lumina_2

        model = lumina_2.Lumina2Pipeline(config)
    elif model_type == 'wan':
        from models import wan

        model = wan.WanPipeline(config)
    elif model_type == 'chroma':
        from models import chroma

        model = chroma.ChromaPipeline(config)
    elif model_type == 'hidream':
        from models import hidream

        model = hidream.HiDreamPipeline(config)
    elif model_type == 'sd3':
        from models import sd3

        model = sd3.SD3Pipeline(config)
    elif model_type == 'cosmos_predict2':
        from models import cosmos_predict2

        model = cosmos_predict2.CosmosPredict2Pipeline(config)
    elif model_type == 'omnigen2':
        from models import omnigen2

        model = omnigen2.OmniGen2Pipeline(config)
    else:
        raise NotImplementedError(f'Model type {model_type} is not implemented')

    # import sys, PIL
    # test_image = sys.argv[1]
    # with torch.no_grad():
    #     vae = model.get_vae().to('cuda')
    #     latents = dataset.encode_pil_to_latents(PIL.Image.open(test_image), vae)
    #     pil_image = dataset.decode_latents_to_pil(latents, vae)
    #     pil_image.save('test.jpg')
    # quit()

    with open(config['dataset']) as f:
        dataset_config = toml.load(f)
    gradient_release = config['optimizer'].get('gradient_release', False)
    ds_section = config['deepspeed']
    ds_config = {
        'train_micro_batch_size_per_gpu': ds_section.get('train_micro_batch_size_per_gpu', 1),
        'gradient_accumulation_steps': ds_section.get('gradient_accumulation_steps', 1),
        'gradient_clipping': ds_section.get('gradient_clipping', 1.0),
        'steps_per_print': ds_section.get('steps_per_print', 200),
        'zero_optimization': {
            'stage': ds_section.get('zero_optimization_stage', 0),
        },

        'memory_efficient_linear': ds_section.get('memory_efficient_linear', False),
        'overlap_comm': ds_section.get('overlap_comm', False),
        'contiguous_gradients': ds_section.get('contiguous_gradients', False),
        'allgather_bucket_size': ds_section.get('allgather_bucket_size', None),
        'reduce_bucket_size': ds_section.get('reduce_bucket_size', None),
    }
    print("ds_config:", ds_config)
    print("CONFIG KEYS:", list(config.keys()))


    #AUTOCAST LOADING
    if not args.autocast and config.get('model', {}).get('dtype'):
        model_dtype = config['model']['dtype']
        dtype_map = {'float32': 'fp32', 'bfloat16': 'bf16', 'float16': 'fp16'}
        autocast_precision = dtype_map.get(model_dtype)
        if autocast_precision and autocast_precision != 'fp32':
            args.autocast = autocast_precision
            print(f" Auto-detected autocast precision from config: {autocast_precision}")
    caching_batch_size = config.get('caching_batch_size', 1)


    dataset_manager = dataset_util.DatasetManager(model, regenerate_cache=regenerate_cache,
                                                  trust_cache=args.trust_cache, caching_batch_size=caching_batch_size)

    train_data = dataset_util.Dataset(dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
    dataset_manager.register(train_data)
    if VERBOSE:
        # =============================================================================
        print("=== DATASET INFO ===")
        print(f"Dataset config path: {config['dataset']}")
        print(f"Expected repeats: 50")
        # check if dataset loaded correctly
        try:
            print(f"Raw dataset length: {len(train_data.data) if hasattr(train_data, 'data') else 'Unknown'}")
            print(f"Dataset with repeats: {len(train_data) if hasattr(train_data, '__len__') else 'Unknown'}")
            # Try to get first item
            first_item = next(iter(train_data))
            print(f"First item keys: {first_item.keys() if isinstance(first_item, dict) else 'Not a dict'}")

        except Exception as e:
            print(f"Dataset error: {e}")
        print("==========================")
        print(f"=== CONFIG STATS ===")
        print(f"Epochs: {config.get('epochs', 'NOT SET')}")
        print(f"Save every n epochs: {config.get('save_every_n_epochs', 'NOT SET')}")
        print(f"Save every n steps: {config.get('save_every_n_steps', 'NOT SET')}")
        print(f"Max steps: {config.get('max_steps', 'NOT SET')}")
        print("==================")
        # =============================================================================

    eval_data_map = {}
    for i, eval_dataset in enumerate(config['eval_datasets']):
        if type(eval_dataset) == str:
            name = f'eval{i}'
            config_path = eval_dataset
        else:
            name = eval_dataset['name']
            config_path = eval_dataset['config']
        with open(config_path) as f:
            eval_dataset_config = toml.load(f)
        eval_data_map[name] = dataset_util.Dataset(eval_dataset_config, model,
                                                   skip_dataset_validation=args.i_know_what_i_am_doing)
        dataset_manager.register(eval_data_map[name])

    if args.dump_dataset:
        # only works for flux
        import torchvision

        dataset_manager.cache(unload_models=False)
        if is_main_process():
            with torch.no_grad():
                os.makedirs(args.dump_dataset, exist_ok=True)
                vae = model.vae.to('cuda')
                train_data.post_init(
                    0,
                    1,
                    1,
                    1,
                    1,
                )
                for i, item in enumerate(train_data):
                    latents = item['latents']
                    latents = latents / vae.config.scaling_factor
                    if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                        latents = latents + vae.config.shift_factor
                    img = vae.decode(latents.to(vae.device, vae.dtype)).sample.to(torch.float32)
                    img = img.squeeze(0)
                    img = ((img + 1) / 2).clamp(0, 1)
                    pil_img = torchvision.transforms.functional.to_pil_image(img)
                    pil_img.save(args.dump_dataset / f'{i}.png')
                    if i >= 100:
                        break
        dist.barrier()
        quit()

    dataset_manager.cache()
    if args.cache_only:
        quit()

    model.load_diffusion_model()

    if adapter_config := config.get('adapter', None):
        model.configure_adapter(adapter_config)
        is_adapter = True
        if init_from_existing := adapter_config.get('init_from_existing', None):
            model.load_adapter_weights(init_from_existing)
    else:
        is_adapter = False

    # if this is a new run, create a new dir for it
    if not resume_from_checkpoint and is_main_process():
        run_dir = os.path.join(config['output_dir'], datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H-%M-%S'))
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(args.config, run_dir)
        shutil.copy(config['dataset'], run_dir)
        for eval_dataset in config['eval_datasets']:
            shutil.copy(eval_dataset['config'], run_dir)
    # wait for all processes then get the most recent dir (may have just been created)
    dist.barrier()
    if resume_from_checkpoint is True:  # No specific folder provided, use most recent
        run_dir = get_most_recent_run_dir(config['output_dir'])
    elif isinstance(resume_from_checkpoint, str):  # Specific folder provided
        run_dir = os.path.join(config['output_dir'], resume_from_checkpoint)
        if not os.path.exists(run_dir):
            raise ValueError(f"Checkpoint directory {run_dir} does not exist")
    else:  # Not resuming, use most recent (newly created) dir
        run_dir = get_most_recent_run_dir(config['output_dir'])

    # WandB logging
    wandb_enable = config.get('monitoring', {}).get('enable_wandb', False)
    if wandb_enable and is_main_process():
        wandb_api_key = config['monitoring']['wandb_api_key']
        wandb_tracker = config['monitoring']['wandb_tracker_name']
        wandb_run_name = config['monitoring']['wandb_run_name']
        logging_dir = run_dir
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=wandb_tracker,
            name=wandb_run_name,
            config=config,
            dir=logging_dir
        )

# =============================================================================
# DYNAMIC LAYER SWAPPING - by obisin
# =============================================================================


    # create model layers
    layers = model.to_layers()
    if VERBOSE:
        print(f"\n=== LAYER DTYPE INSPECTION ===")
        print(f"Total layers: {len(layers)}")

        for i, layer in enumerate(layers):
            layer_type = type(layer).__name__

            # Get parameter dtypes for this layer
            param_dtypes = set()
            param_count = 0
            for name, param in layer.named_parameters():
                param_dtypes.add(param.dtype)
                param_count += 1

            # Get buffer dtypes
            buffer_dtypes = set()
            buffer_count = 0
            for name, buffer in layer.named_buffers():
                buffer_dtypes.add(buffer.dtype)
                buffer_count += 1

            # Format dtype info
            dtype_info = []
            if param_dtypes:
                dtype_info.append(f"params: {list(param_dtypes)}")
            if buffer_dtypes:
                dtype_info.append(f"buffers: {list(buffer_dtypes)}")

            dtype_str = ", ".join(dtype_info) if dtype_info else "no tensors"

            print(f"  {i:2d}: {layer_type:<20} | {dtype_str}")

        print("===============================\n")




    if args.gpu_layers is not None:
        specified_gpu_layers = set(map(int, args.gpu_layers.split(',')))
        # Always include first and last layer for stability
        specified_gpu_layers.add(0)  # First layer
        specified_gpu_layers.add(len(layers) - 1)  # Last layer
        print(f" Automatically added layers 0 and {len(layers) - 1} for stability")
    else:
        # Use initial/final logic, but ensure at least 1 initial and 1 final
        initial_gpu_layers = max(1, args.initial_gpu_layers)  # At least 1
        final_gpu_layers = max(1, args.final_gpu_layers)  # At least 1

    if args.dynamic_swapping:
        print("Applying ENHANCED smart GPU allocation + dynamic swapping...")
        # Force activation checkpointing - it's required for swapping to work
        config['activation_checkpointing'] = True
        config['reentrant_activation_checkpointing'] = False
        # Strategy: Keep first and last layers on GPU permanently, smart swap the rest


        print(f"Setting up enhanced swapping for {len(layers)} layers...")
        # NEW: Parse gpu_layers argument if provided
        if args.gpu_layers:
            try:
                specified_gpu_layers = set(map(int, args.gpu_layers.split(',')))
                print(f" Using specified GPU layers: {sorted(specified_gpu_layers)}")

                # Validate layer indices
                invalid_layers = [idx for idx in specified_gpu_layers if idx >= len(layers) or idx < 0]
                if invalid_layers:
                    raise ValueError(f"Invalid layer indices: {invalid_layers}. Must be 0-{len(layers) - 1}")

            except ValueError as e:
                print(f" Error parsing --gpu_layers: {e}")
                print("Example usage: --gpu_layers 0,1,2,14,18,19,20,21,22")
                raise e
        else:
            specified_gpu_layers = None
            print(f"Using initial/final layer allocation: {initial_gpu_layers}/{final_gpu_layers}")


        # PHASE 1: Determine which layers go where based on specified layers or initial/final counts
        for i, layer in enumerate(layers):
            if hasattr(layer, 'to'):
                if specified_gpu_layers is not None:
                    # Use specified layer indices for GPU placement
                    if i in specified_gpu_layers:
                        try:
                            layer.to(GPU_DEVICE)
                            gpu_resident_layers.add(i)
                            print(f"Layer {i} (specified) -> GPU permanent")
                        except RuntimeError as e:
                            print(f" CRITICAL: Cannot fit specified layer {i} on GPU!")
                            print(f"GPU memory may be insufficient. Consider removing layer {i} from --gpu_layers")
                            raise e
                    else:
                        # Not in specified list, make swappable
                        layer.to(CPU_DEVICE)
                        cpu_swappable_layers.add(i)
                        print(f"Layer {i} (not specified) -> CPU swappable")

                else:
                    # Use original initial/final logic as fallback
                    if i < initial_gpu_layers:
                        # Initial layers on GPU permanently
                        try:
                            layer.to(GPU_DEVICE)
                            gpu_resident_layers.add(i)
                            print(f"Layer {i} (initial) -> GPU permanent")
                        except RuntimeError as e:
                            print(f"GPU exhausted at layer {i}, moving to CPU with swapping")
                            layer.to(CPU_DEVICE)
                            cpu_swappable_layers.add(i)

                    elif i >= (len(layers) - final_gpu_layers):
                        # Final layers on GPU permanently
                        try:
                            layer.to(GPU_DEVICE)
                            gpu_resident_layers.add(i)
                            print(f"Layer {i} (final) -> GPU permanent")
                        except RuntimeError as e:
                            print(f"CRITICAL: Cannot fit final layer {i} on GPU!")
                            raise e
                    else:
                        # Middle layers on CPU with swapping capability
                        layer.to(CPU_DEVICE)
                        cpu_swappable_layers.add(i)
                        print(f"Layer {i} (middle) -> CPU swappable")

        print(f"✓ {len(gpu_resident_layers)} layers permanently on GPU: {sorted(gpu_resident_layers)}")
        print(f"✓ {len(cpu_swappable_layers)} layers on CPU with smart swapping: {sorted(cpu_swappable_layers)}")

        device_cache = LayerDeviceCache(model, layers)

        if args.cast_target:
            casting_handler.precast_all_layers(cpu_swappable_layers, layers)

        if args.selective_packing:
            print(" Packing CPU-swappable layers for optimized transfers...")
            for i in cpu_swappable_layers:
                try:
                    # Pass pinned flag to PackedCPUBlock
                    packed_layers[i] = PackedCPUBlock(layers[i])
                    print(f"✓ Layer {i} packed: {packed_layers[i].total_elements} elements")
                except Exception as e:
                    print(f" Failed to pack layer {i}: {e}")
        else:
            print(" Selective packing disabled - using direct transfers only")

        print(" Pre-calculating layer sizes...")
        layer_sizes_mb = {}
        for i, layer in enumerate(layers):
            layer_sizes_mb[i] = sum(p.numel() * p.element_size() for p in layer.parameters()) / (1024 * 1024)
            print(f"   Layer {i}: {layer_sizes_mb[i]:.1f}MB")


        if args.cast_target:
            print(f"\n=== LAYER CASTING DTYPES ===")
            print(f"Total layers: {len(layers)}")

            for i, layer in enumerate(layers):
                layer_type = type(layer).__name__

                # Get parameter dtypes for this layer
                param_dtypes = set()
                param_count = 0
                for name, param in layer.named_parameters():
                    param_dtypes.add(param.dtype)
                    param_count += 1

                # Get buffer dtypes
                buffer_dtypes = set()
                buffer_count = 0
                for name, buffer in layer.named_buffers():
                    buffer_dtypes.add(buffer.dtype)
                    buffer_count += 1

                # Format dtype info
                dtype_info = []
                if param_dtypes:
                    dtype_info.append(f"params: {list(param_dtypes)}")
                if buffer_dtypes:
                    dtype_info.append(f"buffers: {list(buffer_dtypes)}")

                dtype_str = ", ".join(dtype_info) if dtype_info else "no tensors"

                print(f"  {i:2d}: {layer_type:<20} | {dtype_str}")

            print("===============================\n")


        # PHASE 2: Define the enhanced swapping function
        def add_smart_swapping_to_layer(layer, layer_idx, layers, gpu_resident_layers, cpu_swappable_layers):
            """Add swapping capability with backward pass awareness"""

            original_forward = layer.forward

            # Global state to track backward pass
            if not hasattr(add_smart_swapping_to_layer, 'backward_gpu_layers'):
                add_smart_swapping_to_layer.backward_gpu_layers = set()
                add_smart_swapping_to_layer.forward_active = True

            if not hasattr(add_smart_swapping_to_layer, 'layer_compute_times'):
                add_smart_swapping_to_layer.layer_compute_times = []

            # THREADING: Initialize threading components (once)
            if not hasattr(add_smart_swapping_to_layer, 'threading_initialized'):
                add_smart_swapping_to_layer.threading_initialized = True
                add_smart_swapping_to_layer.gpu_lock = threading.Lock()
                add_smart_swapping_to_layer.current_layer_idx = 0
                add_smart_swapping_to_layer.is_backward_pass = False
                add_smart_swapping_to_layer.training_active = True
                add_smart_swapping_to_layer.background_thread = None

            # CUDA STREAMS: Initialize streams (once) - only if enabled
            if args.cuda_streams and not hasattr(add_smart_swapping_to_layer, 'streams_initialized'):
                add_smart_swapping_to_layer.streams_initialized = True
                add_smart_swapping_to_layer.copy_stream = torch.cuda.Stream()
                add_smart_swapping_to_layer.compute_stream = torch.cuda.Stream()
                if args.async_zero_grad:
                    add_smart_swapping_to_layer.async_zero_stream = torch.cuda.Stream()
                    print(" CUDA Streams enabled with async gradient zeroing")
                else:
                    print(" CUDA Streams enabled for copy-compute overlap")


            # THREADING: Start background thread (once)
            if args.threading and add_smart_swapping_to_layer.background_thread is None:
                add_smart_swapping_to_layer.background_thread = threading.Thread(
                    target=background_layer_manager,
                    daemon=True
                )
                add_smart_swapping_to_layer.background_thread.start()

            @wraps(original_forward)
            def swapped_forward(x, *fwd_args, **kwargs):

                # print_gpu_layer_state("POST-COMPUTE", max((layer_idx - 1), 0), cpu_swappable_layers, layers)

                # THREADING: Update current layer for background thread
                if args.threading:
                    add_smart_swapping_to_layer.current_layer_idx = layer_idx

                # TRACK: What's on GPU when we start
                # print_gpu_layer_state("START", layer_idx, cpu_swappable_layers, layers)

                # 1. INITIALIZATION - Detect backward pass
                if args.threading:
                    # frame_info = [(frame.function, frame.filename) for frame in inspect.stack()]
                    # is_recomputation = (
                    #         any('recompute' in name.lower() for name, _ in frame_info) or
                    #         any('backward' in name.lower() for name, _ in frame_info) or
                    #         any('autograd' in filename.lower() for _, filename in frame_info))

                    # start_time = time.time()
                    is_recomputation = in_recompute(6)
                    # end_time = time.time()
                    # transfer_time = end_time - start_time
                    # print(f" is_recomputation: {transfer_time:.6f}s")

                else:
                    # is_recomputation = False  # Simple detection when not using threading
                    is_recomputation = torch.is_grad_enabled()
                    # is_recomputation = getattr(torch.autograd.grad_mode, '_enabled', True)

                if PROFILE: nvtx.range_push(f"Layer_{layer_idx}_{'Backward' if is_recomputation else 'Forward'}")
                try:
                    # THREADING: Update backward pass status
                    if args.threading:
                        add_smart_swapping_to_layer.is_backward_pass = is_recomputation

                    # Initialize global flags if they don't exist
                    if not hasattr(add_smart_swapping_to_layer, 'current_forward_logged'):
                        add_smart_swapping_to_layer.current_forward_logged = False
                        add_smart_swapping_to_layer.current_backward_logged = False

                    # Summary logging at the start - only once per pass type
                    if not is_recomputation and not add_smart_swapping_to_layer.current_forward_logged:
                        if VERBOSE:
                            print(f" Forward pass: layers {min(cpu_swappable_layers)}-{max(cpu_swappable_layers)}")
                        add_smart_swapping_to_layer.current_forward_logged = True
                        add_smart_swapping_to_layer.current_backward_logged = False
                    elif is_recomputation and not add_smart_swapping_to_layer.current_backward_logged:
                        if VERBOSE:
                            print(f" Backward pass: recomputing layers {max(cpu_swappable_layers)}-{min(cpu_swappable_layers)}")
                        add_smart_swapping_to_layer.current_backward_logged = True
                        add_smart_swapping_to_layer.current_forward_logged = False

                    if not hasattr(add_smart_swapping_to_layer, 'stats_initialized'):
                        add_smart_swapping_to_layer.stats_initialized = True
                        add_smart_swapping_to_layer.prefetch_hits = 0
                        add_smart_swapping_to_layer.prefetch_misses = 0

                    # if layer_idx % 20 == 0:  # Track ram
                    #     ram = psutil.virtual_memory()
                    #     print(f"   💾 Backward Layer {layer_idx}: RAM {ram.percent:.1f}% ({ram.used / 1e9:.1f}GB)")

                    # 2. SMART LAYER MANAGEMENT
                    if layer_idx in cpu_swappable_layers:
                        if PROFILE: nvtx.range_push(f"Smart_Management_L{layer_idx}")
                        try:
                            current_device = device_cache.get_device(layer_idx)
                            layer_already_on_gpu = (current_device.type == 'cuda')

                            if not layer_already_on_gpu:
                                # THREADING: If threading enabled, wait briefly for background thread
                                if args.threading:
                                    print(f" Layer {layer_idx} not ready, waiting for background thread...")
                                    for _ in range(50):
                                        time.sleep(0.000001) #do not touch. very needed for sync
                                        if device_cache.get_device(layer_idx).type == 'cuda':
                                            layer_already_on_gpu = True
                                            print(f" Background thread caught up!")
                                            break

                                # If still not ready, run fallback
                                if not layer_already_on_gpu:

                                    # THREADING: Use lock to prevent race condition
                                    if args.threading:
                                        with add_smart_swapping_to_layer.gpu_lock:
                                            needed_layers = calculate_needed_layers(layer_idx, is_recomputation, args.prefetch)

                                            if args.selective_packing:
                                                cleaned = cleanup_excess_layers_packed(needed_layers, args.selective_packing)
                                                fetched = fetch_missing_layers_packed(needed_layers, args.selective_packing)
                                            else:
                                                cleaned = cleanup_excess_layers(needed_layers)
                                                fetched = fetch_missing_layers(needed_layers)
                                    else:
                                        # No threading, run normally
                                        needed_layers = calculate_needed_layers(layer_idx, is_recomputation, args.prefetch)

                                        if args.selective_packing:

                                            cleaned = cleanup_excess_layers_packed(needed_layers, args.selective_packing)
                                            fetched = fetch_missing_layers_packed(needed_layers, args.selective_packing)
                                        else:
                                            cleaned = cleanup_excess_layers(needed_layers)
                                            fetched = fetch_missing_layers(needed_layers)
                        finally:
                            if PROFILE: nvtx.range_pop()

                        if layer_already_on_gpu:
                            add_smart_swapping_to_layer.prefetch_hits += 1
                        else:
                            add_smart_swapping_to_layer.prefetch_misses += 1

                        device = device_cache.get_device(layer_idx)
                        gpu_success = device.type == 'cuda'
                    else:
                        # Layer not in cpu_swappable_layers (permanent resident)
                        device = device_cache.get_device(layer_idx)
                        gpu_success = device.type == 'cuda'
                        print(f"    Permanent resident layer {layer_idx} on {device}")

                    # Handle GPU failure case
                    if not gpu_success:
                        print(f" Layer {layer_idx} failed to be on GPU, forcing aggressive cleanup...")
                        # Nuclear cleanup - evict ALL layers except this one
                        for cleanup_idx in cpu_swappable_layers:
                            if cleanup_idx != layer_idx and cleanup_idx < len(layers):
                                try:
                                    layers[cleanup_idx].to('cpu')
                                except:
                                    pass


                        gc.collect()
                        torch.cuda.empty_cache()

                        # Try again after cleanup
                        try:
                            layer.to(GPU_DEVICE)
                            device = torch.device(GPU_DEVICE)
                            gpu_success = True
                            print(f" Layer {layer_idx} moved to GPU after aggressive cleanup")
                        except RuntimeError:
                            print(f" CRITICAL: Layer {layer_idx} cannot fit on GPU, skipping computation!")
                            return x  # Pass input through unchanged

                    # if layer_idx % 20 == 0 and (add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses) > 0:
                    #     hit_rate = add_smart_swapping_to_layer.prefetch_hits / (add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses) * 100
                    #     print(f" Prefetch hit rate: {hit_rate:.1f}% ({add_smart_swapping_to_layer.prefetch_hits}/{add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses})")

                    # TRACK: What's on GPU before computation
                    # print_gpu_layer_state("PRE-COMPUTE", layer_idx, cpu_swappable_layers, layers)

                    # 3. MOVE TENSORS TO COMPUTATION DEVICE
                    def move_to_device(tensor, target_device):
                        if hasattr(tensor, 'device') and tensor.device != target_device:
                            return tensor.to(target_device)
                        return tensor

                    x = move_to_device(x, device)
                    new_kwargs = {k: move_to_device(v, device) for k, v in kwargs.items()}
                    new_args = tuple(move_to_device(arg, device) for arg in fwd_args)

                    # Wait for any pending GPU transfers before executing the layer
                    if hasattr(layer, '_packed_block'):
                        for event in layer._packed_block.gpu_events.values():
                            torch.cuda.current_stream().wait_event(event)


                    # 4. COMPUTATION
                    if PROFILE: nvtx.range_push(f"Compute_L{layer_idx}")
                    try:
                        if args.cuda_streams and hasattr(add_smart_swapping_to_layer, 'compute_stream'):
                            with torch.cuda.stream(add_smart_swapping_to_layer.compute_stream):
                                if VERBOSE:
                                    layer_compute_start = time.time()

                                if args.autocast and args.autocast != 'fp32':
                                    if args.autocast in ['f8_e4m3', 'f8_e5m2']:
                                        # Use Transformer Engine for FP8
                                        import transformer_engine.pytorch as te
                                        with te.fp8_autocast():
                                            result = original_forward(x, *tuple(new_args), **new_kwargs)
                                    else:
                                        # Regular PyTorch autocast for FP16/BF16
                                        dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16}
                                        autocast_dtype = dtype_map[args.autocast]
                                        with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                                            result = original_forward(x, *tuple(new_args), **new_kwargs)
                                else:
                                    result = original_forward(x, *tuple(new_args), **new_kwargs)

                                if VERBOSE:
                                    layer_compute_end = time.time()
                                    layer_compute_time = layer_compute_end - layer_compute_start
                                    add_smart_swapping_to_layer.layer_compute_times.append(layer_compute_time)
                                return result
                        else:
                            if VERBOSE:
                                layer_compute_start = time.time()

                            if args.autocast and args.autocast != 'fp32':
                                if args.autocast in ['f8_e4m3', 'f8_e5m2']:
                                    # Use Transformer Engine for FP8
                                    import transformer_engine.pytorch as te
                                    with te.fp8_autocast():
                                        result = original_forward(x, *tuple(new_args), **new_kwargs)
                                else:
                                    # Regular PyTorch autocast for FP16/BF16
                                    dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16}
                                    autocast_dtype = dtype_map[args.autocast]
                                    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                                        result = original_forward(x, *tuple(new_args), **new_kwargs)
                            else:
                                result = original_forward(x, *tuple(new_args), **new_kwargs)

                        if VERBOSE:
                            layer_compute_end = time.time()
                            layer_compute_time = layer_compute_end - layer_compute_start
                            add_smart_swapping_to_layer.layer_compute_times.append(layer_compute_time)

                        return result
                    finally:
                        if PROFILE: nvtx.range_pop()
                finally:
                    if PROFILE: nvtx.range_pop()

            # Replace both forward and __call__ methods
            original_call = layer.__call__

            def swapped_call(self, *args, **kwargs):
                return swapped_forward(*args, **kwargs)

            layer.forward = swapped_forward
            layer.__call__ = swapped_call.__get__(layer, layer.__class__)
            print(f" Layer {layer_idx} wrapped with reorganized smart swapping")

        # After the main training loop:
        for layer_idx in cpu_swappable_layers:
            add_smart_swapping_to_layer(
                layers[layer_idx],
                layer_idx,
                layers,
                gpu_resident_layers,
                cpu_swappable_layers)

        if VERBOSE:
            print("Adding compute timing to permanent GPU layers...")
        for layer_idx in gpu_resident_layers:
            layer = layers[layer_idx]
            original_forward = layer.forward

            if args.compile:
                # Try torch.compile version
                @torch.compile
                def compiled_forward(inputs_tuple, *layer_args, **layer_kwargs):
                    return original_forward(inputs_tuple, *layer_args, **layer_kwargs)


                def create_resident_forward(layer_idx, original_forward, compiled_fn):
                    @wraps(original_forward)
                    def resident_forward(inputs_tuple, *args, **kwargs):
                        if VERBOSE:
                            layer_compute_start = time.time()

                        try:
                            result = compiled_fn(inputs_tuple, *args, **kwargs)
                        except Exception as e:
                            print(f" Compile failed for layer {layer_idx}: {e}")
                            result = original_forward(inputs_tuple, *args, **kwargs)

                        if VERBOSE:
                            layer_compute_end = time.time()
                            layer_compute_time = layer_compute_end - layer_compute_start
                            add_smart_swapping_to_layer.layer_compute_times.append(layer_compute_time)
                        return result

                    return resident_forward


                layer.forward = create_resident_forward(layer_idx, original_forward, compiled_forward)
                if VERBOSE:
                    print(f" Added compiled timing to permanent GPU layer {layer_idx}")

            else:
                # Original working version (no compile)
                def create_resident_forward(layer_idx, original_forward):
                    @wraps(original_forward)
                    def resident_forward(inputs_tuple, *args, **kwargs):
                        if VERBOSE:
                            layer_compute_start = time.time()
                        result = original_forward(inputs_tuple, *args, **kwargs)
                        if VERBOSE:
                            layer_compute_end = time.time()
                            layer_compute_time = layer_compute_end - layer_compute_start
                            add_smart_swapping_to_layer.layer_compute_times.append(layer_compute_time)
                        return result

                    return resident_forward


                layer.forward = create_resident_forward(layer_idx, original_forward)
                if VERBOSE:
                    print(f"✓ Added timing to permanent GPU layer {layer_idx}")

        # PHASE 4: Safety patches for remaining device mismatches
        def patch_tensor_operations():
            """Safety patches for any remaining device mismatches"""
            original_add = torch.Tensor.__add__

            def safe_add(self, other):
                if torch.is_tensor(other) and self.device != other.device:
                    other = other.to(self.device)
                return original_add(self, other)

            torch.Tensor.__add__ = safe_add
            original_mul = torch.Tensor.__mul__

            def safe_mul(self, other):
                if torch.is_tensor(other) and self.device != other.device:
                    other = other.to(self.device)
                return original_mul(self, other)

            torch.Tensor.__mul__ = safe_mul
            print(" Applied safety tensor patches")


        patch_tensor_operations()
        # PHASE 5: Print final status
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(0) / 1e9
            print(f" Enhanced swapping ready! GPU memory: {gpu_memory:.2f}GB")
            print(f"  Strategy: {len(gpu_resident_layers)} permanent + {len(cpu_swappable_layers)} smart swapping")

            def print_swap_stats():
                print(f"Swap stats: {swap_stats['to_gpu']} to GPU, {swap_stats['to_cpu']} to CPU")
                if hasattr(add_smart_swapping_to_layer, 'backward_gpu_layers'):
                    print(f"Currently in backward: {len(add_smart_swapping_to_layer.backward_gpu_layers)} layers on GPU")

    additional_pipeline_module_kwargs = {}
    activation_checkpointing = config['activation_checkpointing']
    if activation_checkpointing:
        if activation_checkpointing == True:
            from functools import partial

            checkpoint_func = partial(torch.utils.checkpoint.checkpoint,
                                      use_reentrant=config['reentrant_activation_checkpointing'])
        elif activation_checkpointing == 'unsloth':
            checkpoint_func = unsloth_checkpoint
        else:
            raise NotImplementedError(f'activation_checkpointing={activation_checkpointing} is not implemented')

        additional_pipeline_module_kwargs.update({
            'activation_checkpoint_interval': config.get('activation_checkpoint_interval', 1),
            'checkpointable_layers': model.checkpointable_layers,
            'activation_checkpoint_func': checkpoint_func,
        })

    num_stages = config.get('pipeline_stages', 1)
    partition_method = config.get('partition_method', 'parameters')
    partition_split = config.get('partition_split', None)


    print("Applying monkey patch to prevent DeepSpeed from forcing GPU placement...")
    # Monkey patch the PipelineModule.__init__ to skip the automatic .to() call
    import deepspeed.runtime.pipe.module as ds_pipe_mod

    original_pipeline_init = ds_pipe_mod.PipelineModule.__init__


    def patched_pipeline_init(self, *args, **kwargs):
        # Call original init but skip the problematic .to() call
        # We'll temporarily replace .to() with a dummy during init
        original_to = self.to
        self.to = lambda *args, **kwargs: self  # Dummy .to() that does nothing
        try:
            original_pipeline_init(self, *args, **kwargs)
        finally:
            # Restore original .to() method after init
            self.to = original_to


    ds_pipe_mod.PipelineModule.__init__ = patched_pipeline_init
    print(" DeepSpeed PipelineModule patched to preserve layer placement")

    try:
        pipeline_model = ManualPipelineModule(
            layers=layers,
            num_stages=num_stages,
            partition_method=partition_method,
            manual_partition_split=partition_split,
            loss_fn=model.get_loss_fn(),
            **additional_pipeline_module_kwargs
        )

    except Exception as e:
        print(f"Error creating ManualPipelineModule: {e}")
        print(f"Error type: {type(e)}")

        # If it's a CUDA error, try to get more info
        if "CUDA" in str(e):
            print("CUDA error detected. Checking CUDA state...")
            if torch.cuda.is_available():
                try:
                    # Try a simple CUDA operation
                    test_tensor = torch.tensor([1.0]).cuda()
                    print("CUDA basic operation works")
                except Exception as cuda_e:
                    print(f"CUDA basic operation failed: {cuda_e}")
            else:
                print("CUDA not available!")

        raise e

    # Restore original methods after pipeline creation
    # if original_pipeline_to is not None:
    #     import deepspeed.runtime.pipe.module as ds_pipe_mod
    #
    #     ds_pipe_mod.PipelineModule.to = original_pipeline_to
        # print("DeepSpeed pipeline methods restored")

    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

    if config['compile']:
        pipeline_model.compile()

    def get_optimizer(model_parameters):
        if len(model_parameters) == 0:
            return DummyOptimizer()

        optim_config = config['optimizer']
        optim_type = optim_config['type']
        optim_type_lower = optim_type.lower()

        args = []
        kwargs = {k: v for k, v in optim_config.items() if k not in ['type', 'gradient_release']}

        if optim_type_lower == 'adamw':
            # TODO: fix this. I'm getting "fatal error: cuda_runtime.h: No such file or directory"
            # when Deepspeed tries to build the fused Adam extension.
            # klass = deepspeed.ops.adam.FusedAdam
            klass = torch.optim.AdamW
        elif optim_type_lower == 'fusedadam':  # ADD THIS
            # import deepspeed
            klass = deepspeed.ops.adam.FusedAdam
        elif optim_type_lower == 'adamw8bit':
            import bitsandbytes
            klass = bitsandbytes.optim.AdamW8bit
        elif optim_type_lower == 'adamw_optimi':
            import optimi
            klass = optimi.AdamW
        elif optim_type_lower == 'stableadamw':
            import optimi
            klass = optimi.StableAdamW
        elif optim_type_lower == 'sgd':
            klass = torch.optim.SGD
        elif optim_type_lower == 'adamw8bitkahan':
            from optimizers import adamw_8bit
            klass = adamw_8bit.AdamW8bitKahan
        elif optim_type_lower == 'offload':
            from torchao.prototype.low_bit_optim import CPUOffloadOptimizer
            klass = CPUOffloadOptimizer
            args.append(torch.optim.AdamW)
            kwargs['fused'] = True
        elif optim_type_lower == 'automagic':
            from optimizers import automagic
            klass = automagic.Automagic
        elif optim_type_lower == 'genericoptim':
            from optimizers import generic_optim
            klass = generic_optim.GenericOptim
        else:
            import pytorch_optimizer
            klass = getattr(pytorch_optimizer, optim_type)

        if optim_config.get('gradient_release', False):
            # Prevent deepspeed from logging every single param group lr
            def _report_progress(self, step):
                lr = self.get_lr()
                mom = self.get_mom()
                deepspeed.utils.logging.log_dist(f"step={step}, skipped={self.skipped_steps}, lr={lr[0]}, mom={mom[0]}",
                                                 ranks=[0])

            deepspeed.runtime.engine.DeepSpeedEngine._report_progress = _report_progress

            # Deepspeed executes all the code to reduce grads across data parallel ranks even if the DP world size is 1.
            # As part of this, any grads that are None are set to zeros. We're doing gradient release to save memory,
            # so we have to avoid this.
            def _exec_reduce_grads(self):
                assert self.mpu.get_data_parallel_world_size() == 1, 'When using gradient release, data parallel world size must be 1. Make sure pipeline_stages = num_gpus.'
                return

            deepspeed.runtime.pipe.engine.PipelineEngine._INSTRUCTION_MAP[
                deepspeed.runtime.pipe.schedule.ReduceGrads] = _exec_reduce_grads

            # When pipelining multiple forward and backward passes, normally updating the parameter in-place causes an error when calling
            # backward() on future micro-batches. But we can modify .data directly so the autograd engine doesn't detect in-place modifications.
            # TODO: this is unbelievably hacky and not mathematically sound, I'm just seeing if it works at all.
            def add_(self, *args, **kwargs):
                self.data.add_(*args, **kwargs)

            for p in model_parameters:
                p.add_ = add_.__get__(p)

            if 'foreach' in inspect.signature(klass).parameters:
                kwargs['foreach'] = False

            # We're doing an optimizer step for each micro-batch. Scale momentum and EMA betas so that the contribution
            # decays at the same rate it would if we were doing one step per batch like normal.
            # Reference: https://alexeytochin.github.io/posts/batch_size_vs_momentum/batch_size_vs_momentum.html
            gas = ds_config['gradient_accumulation_steps']
            if 'betas' in kwargs:
                for i in range(len(kwargs['betas'])):
                    kwargs['betas'][i] = kwargs['betas'][i] ** (1 / gas)
            if 'momentum' in kwargs:
                kwargs['momentum'] = kwargs['momentum'] ** (1 / gas)

            optimizer_dict = {}
            for pg in model.get_param_groups(model_parameters):
                param_kwargs = kwargs.copy()
                if isinstance(pg, dict):
                    # param group
                    for p in pg['params']:
                        param_kwargs['lr'] = pg['lr']
                        optimizer_dict[p] = klass([p], **param_kwargs)
                else:
                    # param
                    optimizer_dict[pg] = klass([pg], **param_kwargs)

            def optimizer_hook(p):
                optimizer_dict[p].step()
                optimizer_dict[p].zero_grad()

            for p in model_parameters:
                p.register_post_accumulate_grad_hook(optimizer_hook)

            from optimizers import gradient_release
            return gradient_release.GradientReleaseOptimizerWrapper(list(optimizer_dict.values()))
        elif optim_type_lower == 'genericoptim':
            kwargs['compile'] = config['compile']
            new_param_groups = []
            param_groups = model.get_param_groups(model_parameters)
            for pg in param_groups:
                params = pg.pop('params')
                params_2d = []
                params_other = []
                for p in params:
                    if p.ndim == 2:
                        params_2d.append(p)
                    else:
                        params_other.append(p)
                pg_2d = pg.copy()
                pg_2d['params'] = params_2d
                if kwargs.get('second_moment_type', None) == 'sn':
                    pg_2d['subset_size'] = 'heuristics'
                for key in ('rank', 'proj_type', 'update_proj_gap'):
                    if key in kwargs:
                        pg_2d[key] = kwargs.pop(key)
                new_param_groups.append(pg_2d)
                pg_other = pg
                pg_other['params'] = params_other
                new_param_groups.append(pg_other)
            return klass(new_param_groups, *args, **kwargs)
        else:
            param_groups = model.get_param_groups(model_parameters)
            return klass(param_groups, *args, **kwargs)

    print("Setting up additional DeepSpeed patches for CPU offloading...")
    # Temporarily disable the pipeline model's to() method during DeepSpeed initialization
    original_pipeline_to = pipeline_model.to


    def dummy_pipeline_to(*args, **kwargs):
        print(f"Blocked pipeline_model.to() call with args: {args}")
        return pipeline_model

    pipeline_model.to = dummy_pipeline_to
    print("Pipeline model to() method temporarily disabled")

    try:
        print("Initializing DeepSpeed...")
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=pipeline_model,
            model_parameters=parameters_to_train,
            optimizer=get_optimizer,
            config=ds_config,
            config_params=ds_config
        )

        print("DeepSpeed initialization completed!")
        print(" Adding timing hooks to DeepSpeed...")

        # 1. Hook optimizer step
        original_optimizer_step = model_engine.optimizer.step


        def timed_optimizer_step(self, closure=None):  # Add 'self' parameter
            if VERBOSE:
                optimizer_start = time.time()
            result = original_optimizer_step(closure)
            if VERBOSE:
                optimizer_end = time.time()
                optimizer_time = optimizer_end - optimizer_start

            if not hasattr(model_engine, 'optimizer_times'):
                model_engine.optimizer_times = []

            if VERBOSE:
                model_engine.optimizer_times.append(optimizer_time)
                print(f" Optimizer step: {optimizer_time:.4f}s")
            return result


        # Properly bind the method
        import types

        model_engine.optimizer.step = types.MethodType(timed_optimizer_step, model_engine.optimizer)

        # 2. Try to hook backward (might not exist)
        if hasattr(model_engine, 'backward'):
            original_backward = model_engine.backward


            def timed_backward(self, loss):  # Add 'self' parameter
                if VERBOSE:
                    backward_start = time.time()
                result = original_backward(loss)
                if VERBOSE:
                    backward_end = time.time()
                    backward_time = backward_end - backward_start

                if not hasattr(model_engine, 'backward_times'):
                    model_engine.backward_times = []

                if VERBOSE:
                    model_engine.backward_times.append(backward_time)
                    print(f" Backward pass: {backward_time:.4f}s")
                return result


            model_engine.backward = types.MethodType(timed_backward, model_engine)
            print(" Hooked backward pass")
        else:
            print(" No backward method found")

        print(" DeepSpeed timing hooks installed")

    except Exception as e:
        print(f"DeepSpeed initialization error: {e}")
        raise e

    finally:
        # Restore the original to() method after DeepSpeed initialization
        pipeline_model.to = original_pipeline_to
        print("Pipeline model to() method restored")

    # ds_config['zero_optimization'] = ds_config.get('zero_optimization', {})
    # ds_config['zero_optimization']['stage'] = 0
    # assert int(os.getenv('WORLD_SIZE', '1')) == 1, 'This trainer is single-GPU only for now.'
    model_engine.allreduce_gradients = lambda *args, **kwargs: None
    print(" Disabled DeepSpeed gradient allreduce for mixed device training")

    mixed_device_clip_grad_norm = get_mixed_precision_clip_function()

    model_engine.clip_fp32_gradients = lambda: mixed_device_clip_grad_norm(
        model_engine.module.parameters(),
        model_engine.gradient_clipping()
    )

    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    if config['warmup_steps'] > 0:
        warmup_steps = config['warmup_steps']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / warmup_steps,
                                                             total_iters=warmup_steps)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler],
                                                             milestones=[warmup_steps])
    model_engine.lr_scheduler = lr_scheduler


    # original_optimizer_step = model_engine.optimizer.step

    def create_conditional_optimizer_patch(model_engine, sync_mode, only_on_resume=False, resumed_from_checkpoint=False):
        """Create device sync patch with conditional logic"""

        # Skip patch entirely if sync is off or if only_on_resume=True and we didn't resume
        if sync_mode == 'off':
            print(" Device sync disabled")
            return


        if not resumed_from_checkpoint:
            print(" Device sync skipped (fresh training, efficiency mode)")
            return

        original_optimizer_step = model_engine.optimizer.step

        def patched_optimizer_step(self, closure=None):
            # Smart device sync with detailed logging
            needs_sync = False
            device_issues = []
            grad_mismatches = 0
            state_mismatches = 0

            # Check for ANY device mismatches (gradients OR optimizer states)
            for param_group in self.param_groups:
                for param in param_group['params']:
                    # Check gradient mismatch
                    if param.grad is not None and param.grad.device != param.device:
                        needs_sync = True
                        grad_mismatches += 1
                        device_issues.append(f"grad mismatch: {param.grad.device} vs {param.device}")
                        break

                    # Check optimizer state mismatch
                    if param in self.state:
                        param_state = self.state[param]
                        for key, value in param_state.items():
                            if torch.is_tensor(value) and value.device != param.device:
                                needs_sync = True
                                state_mismatches += 1
                                device_issues.append(f"{key} mismatch: {value.device} vs {param.device}")
                                break
                        if needs_sync:
                            break
                if needs_sync:
                    break

            if needs_sync:
                print(f" Device sync needed: {device_issues[0]}")

                synced_grads = 0
                synced_states = 0

                # Fix ALL mismatches in one go
                for param_group in self.param_groups:
                    for param in param_group['params']:
                        # Sync gradients to parameter device
                        if param.grad is not None and param.grad.device != param.device:
                            param.grad.data = param.grad.data.to(param.device)
                            synced_grads += 1

                        # Sync optimizer states to parameter device
                        if param in self.state:
                            param_state = self.state[param]
                            for key, value in param_state.items():
                                if torch.is_tensor(value) and value.device != param.device:
                                    param_state[key] = value.to(param.device)
                                    synced_states += 1

                print(f"   Device sync: moved {synced_grads} grads, {synced_states} optimizer states")
            else:
                print(" Device sync skipped (no mismatch detected)")

            return original_optimizer_step(closure)

        # Apply the patch

        model_engine.optimizer.step = types.MethodType(patched_optimizer_step, model_engine.optimizer)
        print(f" Patched optimizer step for {sync_mode} device sync")


    model.model_engine = model_engine

    if model_engine.is_pipe_parallel:
        grid = model_engine.grid
        model_engine.first_last_stage_group = dist.new_group(ranks=[grid.pp_group[0], grid.pp_group[-1]])

    train_data.post_init(
        model_engine.grid.get_data_parallel_rank(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
        config.get('image_micro_batch_size_per_gpu', model_engine.train_micro_batch_size_per_gpu()),
    )

    for eval_data in eval_data_map.values():
        eval_data.post_init(
            model_engine.grid.get_data_parallel_rank(),
            model_engine.grid.get_data_parallel_world_size(),
            config.get('eval_micro_batch_size_per_gpu', model_engine.train_micro_batch_size_per_gpu()),
            config['eval_gradient_accumulation_steps'],
            config.get('image_eval_micro_batch_size_per_gpu',
                       config.get('eval_micro_batch_size_per_gpu', model_engine.train_micro_batch_size_per_gpu())),
        )

    # Might be useful because we set things in fp16 / bf16 without explicitly enabling Deepspeed fp16 mode.
    # Unsure if really needed.
    communication_data_type = config['lora']['dtype'] if 'lora' in config else config['model']['dtype']
    model_engine.communication_data_type = communication_data_type

    train_dataloader = dataset_util.PipelineDataLoader(train_data, model_engine,
                                                       model_engine.gradient_accumulation_steps(), model)

    print(f"\n=== DATASET ===")
    print(f"train_data length: {len(train_data)}")
    print(f"train_dataloader length: {len(train_dataloader)}")
    print(f"Micro batch size: {model_engine.train_micro_batch_size_per_gpu()}")
    print(f"Gradient accumulation: {model_engine.gradient_accumulation_steps()}")
    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
    print(f"Steps per epoch:{steps_per_epoch}")
    print(f"Target epochs: {config['epochs']}")
    expected_steps = steps_per_epoch * config['epochs']
    print(f"Expected total steps: {expected_steps}")
    print("===========================================")
    print("\n=== DEEPSPEED CONFIG ===")
    print(f"zero_optimization_stage       : {ds_config['zero_optimization']['stage']}")
    print(f"train_micro_batch_size_per_gpu: {model_engine.train_micro_batch_size_per_gpu()}")
    print(f"gradient_accumulation_steps   : {model_engine.gradient_accumulation_steps()}")
    print(f"gradient_clipping             : {ds_config['gradient_clipping']}")
    print(f"steps_per_print               : {ds_config['steps_per_print']}")
    print(f"memory_efficient_linear       : {ds_config['memory_efficient_linear']}")
    print(f"overlap_comm                  : {ds_config['overlap_comm']}")
    print(f"contiguous_gradients          : {ds_config['contiguous_gradients']}")
    print(f"allgather_bucket_size         : {ds_config['allgather_bucket_size']}")
    print(f"reduce_bucket_size            : {ds_config['reduce_bucket_size']}")
    print("===========================================")

    step = 1

    if resume_from_checkpoint:
        load_path, client_state = model_engine.load_checkpoint(
            run_dir,
            load_module_strict=False,
            load_lr_scheduler_states='force_constant_lr' not in config,
        )

        resumed_from_checkpoint = load_path is not None

        if resumed_from_checkpoint:
            print(" Fixing tensor device placement after checkpoint load...")

            # Force all gradients to match their parameter devices
            for param in model_engine.module.parameters():
                if param.requires_grad and param.grad is not None:
                    if param.grad.device != param.device:
                        param.grad.data = param.grad.data.to(param.device)

            print(" Gradient device placement fixed")

            moved_params = 0
            moved_grads = 0
            moved_states = 0

            # 1. Ensure all model parameters are on correct devices
            for param in model_engine.module.parameters():
                if param.requires_grad:
                    target_device = GPU_DEVICE if not args.dynamic_swapping else param.device

                    if param.device != target_device:
                        param.data = param.data.to(target_device)
                        moved_params += 1

                    if param.grad is not None and param.grad.device != param.device:
                        param.grad.data = param.grad.data.to(param.device)
                        moved_grads += 1

            # 2. Fix ALL optimizer states to match their parameters
            for param_group in model_engine.optimizer.param_groups:
                for param in param_group['params']:
                    if param in model_engine.optimizer.state:
                        param_state = model_engine.optimizer.state[param]
                        for key, value in param_state.items():
                            if torch.is_tensor(value) and value.device != param.device:
                                param_state[key] = value.to(param.device)
                                moved_states += 1

            print(f" Checkpoint device fix: {moved_params} params, {moved_grads} grads, {moved_states} states")
            torch.cuda.synchronize()
            # event_based_sync("gpu_transfer", idx) #wait for all ops to sync on restart

        # EARLY SYNC - only if enabled and actually resumed # unstable on 2060 #unstable on 2080
        if args.device_sync_mode == 'early' and resumed_from_checkpoint:
            print(" Post-checkpoint device synchronization (early mode)...")

            moved_params = 0
            moved_grads = 0
            moved_states = 0

            try:
                # Force all model parameters to GPU
                for param in model_engine.module.parameters():
                    try:
                        if param.device.type == 'cpu':
                            param.data = param.data.to(GPU_DEVICE)
                            moved_params += 1
                            if param.grad is not None:
                                param.grad.data = param.grad.data.to(GPU_DEVICE)
                                moved_grads += 1
                    except RuntimeError as e:
                        print(f" Failed to move parameter to GPU: {e}")
                        continue

                # Force all optimizer states to GPU
                for param_group in model_engine.optimizer.param_groups:
                    for param in param_group['params']:
                        if param in model_engine.optimizer.state:
                            state = model_engine.optimizer.state[param]
                            for key, value in state.items():
                                try:
                                    if torch.is_tensor(value) and value.device.type == 'cpu':
                                        state[key] = value.to(GPU_DEVICE)
                                        moved_states += 1
                                except RuntimeError as e:
                                    print(f" Failed to move optimizer state {key}: {e}")
                                    continue

                print(f" Early sync: {moved_params} params, {moved_grads} grads, {moved_states} states")

            except Exception as e:
                print(f" Early device sync failed: {e}")
                print("Falling back to late sync mode")
                args.device_sync_mode = 'late'  # Fallback to late sync

        dist.barrier()
        assert load_path is not None
        train_dataloader.load_state_dict(client_state['custom_loader'])
        step = client_state['step'] + 1

        # del client_state

        if load_path is not None:
            print("Deep checkpoint cleanup...")

            # 1. Clean client_state deeply
            if 'client_state' in locals() and client_state is not None:
                def deep_cleanup(obj):
                    if isinstance(obj, dict):
                        for key in list(obj.keys()):
                            if torch.is_tensor(obj[key]):
                                del obj[key]
                            elif isinstance(obj[key], (dict, list)):
                                deep_cleanup(obj[key])
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            if torch.is_tensor(item):
                                obj[i] = None


                # deep_cleanup(client_state)
                del client_state

            # # 2. Clear PyTorch's internal checkpoint cache
            if hasattr(torch.utils.checkpoint, '_checkpoint_cache'):
                torch.utils.checkpoint._checkpoint_cache.clear()

            # # 3. Force cleanup of GPU memory fragments
            # if step % 50 == 0:
            #     gc.collect()
            #     torch.cuda.empty_cache()

            # 4. Memory verification
            post_cleanup_memory = torch.cuda.memory_allocated(0) / 1e9
            print(f" Memory after checkpoint cleanup: {post_cleanup_memory:.1f}GB")


        if is_main_process():
            print(f'Resuming training from checkpoint. Resuming at epoch: {train_dataloader.epoch}, step: {step}')
    else:
        resumed_from_checkpoint = False

    create_conditional_optimizer_patch(
        model_engine,
        args.device_sync_mode,
        args.sync_only_on_resume,
        resumed_from_checkpoint
    )

    if 'force_constant_lr' in config:
        model_engine.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = config['force_constant_lr']

    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
    model_engine.total_steps = steps_per_epoch * config['epochs']

    eval_dataloaders = {
        # Set num_dataloader_workers=0 so dataset iteration is completely deterministic.
        # We want the exact same noise for each image, each time, for a stable validation loss.
        name: dataset_util.PipelineDataLoader(eval_data, model_engine, config['eval_gradient_accumulation_steps'],
                                              model, num_dataloader_workers=0)
        for name, eval_data in eval_data_map.items()
    }

    epoch = train_dataloader.epoch
    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
    saver = utils.saver.Saver(args, config, is_adapter, run_dir, model, train_dataloader, model_engine, pipeline_model)

    disable_block_swap_for_eval = config.get('disable_block_swap_for_eval', False)



    # TODO: this is state we need to save and resume when resuming from checkpoint. It only affects logging.
    epoch_loss = 0
    num_steps = 0
    while True:
        if PROFILE: nvtx.range_push(f"Training_Step_{step}")
        try:
            # empty_cuda_cache()
            model_engine.reset_activation_shape()

            if (step >= 3 and step < 5) and TRACING_PROFILER:
                print(f" Profiling step {step}...")
                print(f" Saving profile to: {run_dir}/profile_step_{step}.json")

                iterator = get_data_iterator_for_step(train_dataloader, model_engine)  # ← Move here

                with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True,
                        with_stack=True,
                        profile_memory=True
                ) as prof:
                    try:
                        loss = model_engine.train_batch(iterator).item()
                    except Exception as e:
                        print(f" Training step {step} failed: {e}")
                        raise e

                # Save profile
                prof.export_chrome_trace(f"{run_dir}/profile_step_{step}.json")
                print(f" Profile saved to {run_dir}/profile_step_{step}.json")
                print("   Open in chrome at chrome://tracing")



            iterator = get_data_iterator_for_step(train_dataloader, model_engine)

            # if step % 50 == 0:
            #     gc.collect()
            #     torch.cuda.empty_cache()

            if step % 50 == 0:
                print(f"\n=== STEP {step} STATS ===")
                print(f"Current epoch: {epoch}")
                print(f"Target epochs: {config['epochs']}")
                print(f"Dataset epoch: {train_dataloader.epoch}")
                print(f"Steps per epoch: {steps_per_epoch}")
                print(f"Expected total steps: {model_engine.total_steps}")
                if (add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses) > 0:
                    hit_rate = add_smart_swapping_to_layer.prefetch_hits / (
                                add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses) * 100
                    print(f"Prefetch hit rate: {hit_rate:.1f}% ({add_smart_swapping_to_layer.prefetch_hits}/{add_smart_swapping_to_layer.prefetch_hits + add_smart_swapping_to_layer.prefetch_misses})")
                print("========================")

            # Progress monitoring - start of step
            start_time = time.time()
            start_memory = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
            print(f"\n🚀 Starting step {step}...")
            reset_current_step_stats()
            print(f"   Initial GPU memory: {start_memory:.1f}GB")

            if PROFILE: nvtx.range_push("Zero_Gradients")
            async_zero_gradients(model_engine)
            if PROFILE: nvtx.range_pop()

            if PROFILE: nvtx.range_push("Train_Batch")
            try:
                try:
                    loss = model_engine.train_batch(iterator).item()

                    if args.lazy_lora_steps > 1:
                        if PROFILE: nvtx.range_push("Lazy_LoRA_Step")
                        optimizer_stepped = lazy_lora_step(model_engine, step)
                        if PROFILE: nvtx.range_pop()
                    else:
                        # Normal behavior when lazy LoRA is disabled
                        optimizer_stepped = True

                    # Progress monitoring - end of step
                    end_time = time.time()
                    end_memory = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
                    step_duration = end_time - start_time

                    print(f"")
                    print(f" Step {step} completed successfully!")
                    print(f"   Duration: {step_duration:.2f} seconds")
                    print(f"   AVG Layer Duration: {step_duration / steps_per_epoch:.3f} seconds")
                    print(f"   Loss: {loss:.6f}")
                    print(f"   Final GPU memory: {end_memory:.2f}GB")
                    print(f"   Memory change: {end_memory - start_memory:+.2f}GB")

                    # Extract all timing components
                    if hasattr(add_smart_swapping_to_layer, 'layer_compute_times') and VERBOSE:
                        total_layer_compute = sum(add_smart_swapping_to_layer.layer_compute_times)
                        layer_count = len(add_smart_swapping_to_layer.layer_compute_times)
                    else:
                        total_layer_compute = 0.0


                    # Call existing track_step_performance function
                    if VERBOSE:
                        track_step_performance(step_duration, step, total_layer_compute, False)

                except Exception as e:
                    print(f" Training step {step} failed: {e}")
                    print("This might indicate device synchronization issues or GPU instability.")
                    raise e
            finally:
                if PROFILE: nvtx.range_pop()

            epoch_loss += loss
            num_steps += 1
            train_dataloader.sync_epoch()

            new_epoch, checkpointed, saved = saver.process_epoch(epoch, step)
            finished_epoch = True if new_epoch != epoch else False

            if is_main_process() and step % config['logging_steps'] == 0:
                tb_writer.add_scalar(f'train/loss', loss, step)
                if wandb_enable:
                    wandb.log({'train/loss': loss, 'step': step})
                if optimizer.__class__.__name__ == 'Prodigy':
                    prodigy_d = get_prodigy_d(optimizer)
                    tb_writer.add_scalar(f'train/prodigy_d', prodigy_d, step)
                if optimizer.__class__.__name__ == 'Automagic':
                    lrs, avg_lr = _get_automagic_lrs(optimizer)
                    tb_writer.add_histogram(f'train/automagic_lrs', lrs, step)
                    tb_writer.add_scalar(f'train/automagic_avg_lr', avg_lr, step)


            if finished_epoch:
                if is_main_process():
                    tb_writer.add_scalar(f'train/epoch_loss', epoch_loss / num_steps, epoch)
                    if wandb_enable:
                        wandb.log({'train/epoch_loss': epoch_loss / num_steps, 'epoch': epoch})
                epoch_loss = 0
                num_steps = 0
                print_detailed_memory_stats("(Training Start)")
                epoch = new_epoch
                if epoch is None:
                    break

            saver.process_step(step)
            # step += 1

            # Add aggressive cleanup here - after step increment but before memory monitoring
            aggressive_cpu_cleanup()

            # Memory monitoring every 10 steps
            if step % 100 == 0:
                print_detailed_memory_stats(f"(Step {step})")
                # check_wsl_free_memory()

            step += 1

            if config.get('max_steps') and step >= config['max_steps']:
                print(f"Reached max_steps ({config['max_steps']}), stopping training")
                break
        finally:
            if PROFILE: nvtx.range_pop()

    # Training loop has ended - stop background threading
    # Save final training state checkpoint and model, unless we just saved them.
    if args.threading:
        stop_background_threading()

    if not checkpointed:
        saver.save_checkpoint(step)

    if not saved:
        saver.save_model(f'epoch{epoch}')

    if is_main_process():
        print('TRAINING COMPLETE!') #PRINT times etc here and final ststas
        track_step_performance(step_duration, step, total_layer_compute, True)



