# DGLS (Dynamic GPU Layer Swapping) Drop-In/Read Only version

Smart dynamic layer swapping between GPU and CPU for optimal performance with comprehensive mixed precision handling, synchronization control, and copy-compute overlap optimization. Built on top of diffusion-pipe. The primary reqason fo rthis in Linux was to test DeepSpeed features. Most of the code can be easily ported to Windows as its in Pytorch.

## Features

* **Mixed Precision Enabled**: Can cast dtypes and handle mixed precision so you can cast manually or auto
* **Threading and Synchronous Copy-Compute**: Background thread management with overlapped memory transfers and computation
* **Predictive Prefetching**: Intelligently predicts and preloads layers before they're needed whilst maintaining ones needed for next compute
* **Packed Transfers**: Optimizes large layer transfers using contiguous memory packing for faster PCIe bandwidth utilization
* **CUDA Streams**: Uses CUDA streams to overlap memory transfers with computation
* **Forward/Backward Awareness**: Automatically adapts swapping direction based on training phase detection

This is a memory management system that maintains training performance while dramatically reducing VRAM requirements through intelligent layer orchestration.

## Performance

I achieved a 30% speed reduction over a fully on VRAM model with an RTX 2080Ti with a mixed layer model, f32→bf16 precision targeting, prefetch 6, batch layer moving, event sync, and selective packing.

### Real-World Results

**RTX 2060 (6GB)**: Can train 14B parameter model Loras using only 4GB of memory
**RTX 2080 Ti (11GB)**: 14B+ parameter models with optimized settings  
**High-end cards**: 20B+ parameter models with theoretical scaling
**Practical Benefit**: Train much larger models on mid-range cards, and theoretically even larger models on high-end hardware by maximizing VRAM efficiency through dynamic management.

## System Requirements

**Single GPU Only**: Designed for world_size=1 training. You can specify different GPUs if you have multiple cards, but each training session uses only one GPU at a time.

**GPU Compatibility**: Tested on RTX 2060 and RTX 2080 Ti. Other CUDA-capable GPUs should work but may require different optimal settings.

**Memory Requirements**:
- **RAM**: 32GB minimum, 64GB recommended
- **RAM Speed**: 3600MHz+ recommended
- **VRAM**: 6GB+ (enables training models that normally require 24GB+)

**Performance Notes**:
- PCIe bandwidth affects transfer speeds
- RAM speed directly impacts swapping performance
- System can maintain ~70% usability during training

## Installation

### Fork Version

```bash
git clone https://github.com/obisin/dgls
cd dynamic-swapping-trainer-fork
pip install -r requirements.txt
```

### Drop-in Method Version

```bash
# 1. Download the enhancement files
git clone https://github.com/obisin/dgls-method
cd dynamic-swapping-method

# 2. Copy to your existing training project (NO modifications to your code!)
cp dgls_train.py /path/to/your/training/project/
cp utils/mixed_precision_handler.py /path/to/your/training/project/utils/
cp utils/memory_monitor.py /path/to/your/training/project/utils/
```

- **Drop-in files** original files remain completely unchanged
- **Zero modifications** to your existing code


### Dependencies
**Required**:
- Python 3.8+
- PyTorch 2.0+
- DeepSpeed
- CUDA toolkit matching your PyTorch version

## How It Works

DGLS implements a **three-tier memory hierarchy** that intelligently manages layer placement during training.

### Threaded Optimisation

```
Forward Pass:  [GPU: layers 0,1,2] [CPU: layers 3,4,5,6,7]
               ↓ Compute layer 2, prefetch layer 3,4
Next Step:     [GPU: layers 1,2,3,4] [CPU: layers 0,5,6,7]
               ↓ Clean layer 1, Compute layer 3, fetch layer 5
Next Step:     [GPU: layers 2,3,4,5] [CPU: layers 0,1,6,7]
               ↓ Clean layer 2, Compute layer 4, fetch layer 6
Next Step:     [GPU: layers 3,4,5,6] [CPU: layers 0,1,2,7]
                Always staying two layers ahead.
```

**Threading Integration**: best when you increase your batch size, have larger prefetch windows, train with larger models, larger datasets or all combined. This is due to compute time being faster than trasnfers. 
This however enables better training with higher settings to find the optimal balance between compute and copy speeds. 

## Arguments Guide

### Core Swapping Settings

#### `--dynamic_swapping`
**Default**: `True`
**When to use**: Always enabled! use normal train.py to train without it.

#### `--gpu_id`
**Default**: `0`  
**Usage**: Specify which GPU to use (0, 1, 2, etc.)  
**When to use**: Multi-GPU systems where you want to use a specific card
```bash
--gpu_id 1  # Use second GPU
```

### Layer Allocation Strategies

#### `--initial_gpu_layers` & `--final_gpu_layers`
**Default**: `1` each  
**Usage**: Number of layers to keep permanently on GPU at start/end of model  
**When to use**: 
- Higher values (2-10) for more VRAM, better stability
- Keep at 1 for maximum swapping on low VRAM GPUs
```bash
--initial_gpu_layers 3 --final_gpu_layers 2  # 3 initial, 2 final on GPU
```

#### `--gpu_layers`
**Usage**: Comma-separated list of specific layer indices to keep on GPU  
**When to use**: Precise control over which layers stay resident such as with mixed layer models like sdxl  
**Overrides**: initial_gpu_layers and final_gpu_layers when specified
```bash
--gpu_layers "0,1,2,18,19,20,21"  # Specific layers on GPU
```

### Performance Optimization

#### `--prefetch`
**Default**: `0` (disabled)  
**Usage**: Number of layers to prefetch ahead of current computation. You need to calculate the space for the prefecth. With --prefetch 2 you need space for 4 layers to be free on backpass- 2 for the compute and 2 for the prefetch.  
**When to use**: 
- `1`: Single layer prefetch, good for most cases
- `2+`: Multiple layer prefetch, needs more VRAM
- `0`: Disabled
```bash
--prefetch 1  # Prefetch next layer
```

#### `--threading`
**Default**: `False`  
**Usage**: Enable background thread for automatic layer management  
**When to use**: Can improve performance by overlapping transfers with computation. Works best with slower compute so increase batch size, use better models,  
**Caution**: May cause instability on some systems
```bash
--threading  # Enable background threading
```

#### `--cuda_streams`
**Default**: `False`  
**Usage**: Enable CUDA streams for more copy-compute overlap  
**When to use**: More VRAM available, want maximum performance  
**Requires**: Additional VRAM for stream buffers
```bash
--cuda_streams  # Enable CUDA streams
```

#### `--batch_move`
**Default**: `False`  
**Usage**: Move multiple layers simultaneously instead of one-by-one  
**When to use**: GPUs with better bus/copy speeds 30series+ should be better with this
```bash
--batch_move  # Enable batch layer moving
```

### Memory Management

#### `--selective_packing`
**Default**: `0` (disabled)  
**Usage**: Size threshold in MB for using packed transfers  
**When to use**: Optimize transfer speed for large layers
```bash
--selective_packing 64  # Pack layers larger than 64MB, calls layer.to otherwise
```

#### `--async_zero_grad`
**Default**: `False`  
**Usage**: Zero gradients asynchronously on separate CUDA stream  
**When to use**: With CUDA streams for additional overlap
```bash
--async_zero_grad  # Async gradient zeroing
```

### Precision & Casting

#### `--cast_target`
**Usage**: Cast layer dtype at startup. it'll target only layer with a specific dtype and cast that to what you want. It only targets swappable layers, so place sensitive layers on GPU. 
**Format**: `FROM TO`  
**When to use**: Reduce memory usage or optimize for specific hardware
```bash
--cast_target f32 bf16  # Cast float32 to bfloat16
```

#### `--autocast`
**Options**: `fp32`, `fp16`, `bf16`, `f8_e4m3`, `f8_e5m2`  
**Usage**: Cast everything! 
**When to use**: 
- `fp16`/`bf16`: Standard mixed precision
- `f8_*`: Experimental ultra-low precision
```bash
--autocast bf16  # Use bfloat16 autocast
```

#### `--mixed_precision`
**Default**: `auto`  
**Options**: `auto`, `f32`, `bf16`, `f16`  
**Usage**: Target dtype for problematic mixed precision conversions for pytorch problem so you can can handle models, vaes etc of all mixed dtypes.  
**When to use**: `auto` handles most cases automatically

### Device Synchronization

#### `--device_sync_mode`
**Default**: `late`  
**Options**: `early`, `late`, `off`  
**Usage**: When to synchronize devices after checkpoint loading. Best to leave this late in most cases.  
**When to use**:
- `late`: Safest, default option- just leave on late.
- `early`: May be unstable on some GPUs
- `off`: Only for perfect dynamic swapping
```bash
--device_sync_mode late  # Safe device sync
```

#### `--sync_only_on_resume`
**Usage**: Only run device sync when resuming from checkpoint  
**When to use**: Fresh training doesn't need sync, resume does

#### `--lazy_lora_steps`
**Default**: `1`  
**Usage**: Update LoRA adapters every N steps instead of every step  
**When to use**: LoRA training with memory constraints
```bash
--lazy_lora_steps 2  # Update LoRA every 2 steps
```

#### `--compile`
**Default**: `False`  
**Usage**: Use torch.compile optimization for resident GPU layers only 
**When to use**: PyTorch 2.0+ for additional speedup
```bash
--compile  # Enable torch.compile
```

#### `--event_sync`
**Default**: `False`  
**Usage**: Use CUDA events instead of torch.cuda.synchronize(). Better with it on.  
**When to use**: Fine-tuned performance optimization
```bash
--event_sync  # Use CUDA events
```

#### `--verbose`
**Default**: `False`  
**Usage**: Enable verbose output with detailed timing and transfer information  
**When to use**: First use, Debugging performance issues, understanding swapping behavior, or monitoring detailed transfer statistics  
**Output includes**: Layer dtype inspection, transfer timing, memory stats, optimizer timing, prefetch hit rates

## Configuration Examples

DGLS uses TOML configuration files to define model, training, and dataset settings. Here are example configurations for different scenarios:

The Example tomls are in the toml folder.

### Conservative Training (6GB VRAM Safe)
For RTX 2060 or limited VRAM setups - for stability:

I used with this command: 

```bash
CUDA_VISIBLE_DEVICES=0 python dgls_train.py --config conservativelora.toml --dynamic_swapping --initial_gpu_layers 3 --final_gpu_layers 4 --prefetch 1 --event_sync
```

**Training Config** (`conservativelora.toml`):
- Batch size: 1, no gradient accumulation
- LoRA rank: 8 (minimal memory usage)
- Learning rate: 5e-5 (stable)
- Target modules: Core attention only (`to_q`, `to_k`, `to_v`, `to_out.0`)

**Dataset Config** (`dogdataset.toml`):
- 5 images × 50 repeats = 250 training samples
- 512×512 resolution
- Simple "a photo of sks dog sitting" captions

### Aggressive Training (8GB+ VRAM)

I used with this command: 

```bash
CUDA_VISIBLE_DEVICES=0 python dgls_train.py --config aggressivelora.toml --dynamic_swapping --initial_gpu_layers 3 --final_gpu_layers 3 --prefetch 2 --event_sync --threading --batch_move --selective_packing
```

**Training Config** (`aggressivelora.toml`):
- Batch size: 3 per GPU
- LoRA rank: 64 (higher quality)
- Learning rate: 1e-4 (faster training)
- Extended target modules for better coverage

**Dataset Config** (`bengalcatdataset.toml`):
- 50 images × 25 repeats = 1,250 training samples
- Bengal cat specific captions
- Aspect ratio bucketing enabled

### Key TOML Structure

See exmaple toml in toml folder for a better look.

```toml
# Basic training settings
epochs = 2
pipeline_stages = 1
warmup_steps = 3
activation_checkpointing = true

[model]
type = 'wan'  # or 'flux', 'sdxl', etc.
dtype = 'bfloat16'
transformer_dtype = 'float8_e4m3fn'

[adapter]
type = 'lora'
rank = 32  # 8=conservative, 64=aggressive
target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

[optimizer]
type = 'adamw'
lr = 5e-5

[deepspeed]
zero_optimization_stage = 0 #recommended to keep this 0
train_micro_batch_size_per_gpu = 1 # increase to help sync threading
gradient_accumulation_steps = 1
```

### Usage with DGLS

```bash
# Conservative training with swapping
python dgls_train.py --config conservativelora.toml \
    --dynamic_swapping \
    --initial_gpu_layers 1 \
    --final_gpu_layers 1 \
    --prefetch 1

# Aggressive training with optimizations
python dgls_train.py --config aggressivelora.toml \
    --dynamic_swapping \
    --initial_gpu_layers 3 \
    --final_gpu_layers 2 \
    --prefetch 2 \
    --threading \
    --batch_move
```

The TOML files define the model and training parameters, while DGLS arguments control the memory management and swapping behavior.

## Example Commands

They can all be stacked together if you have to try.

Use `--verbose` at first use so you can see how the training runs and adjust settings better.

If you have multi-GPUs I would advise using CUDA_VISIBLE_DEVICES=0 or 1 depending on the GPU you want to use. You can use 0,1 if you only want to swap between GPUS. It cannot handle more than two devices swapping

### Basic Usage (Low VRAM - 6GB)
```bash
python dgls_train.py --config config.toml \
    --dynamic_swapping \
    --initial_gpu_layers 3 \
    --final_gpu_layers 3 \
    --prefetch 1\
    --event_sync
```

### Balanced Performance (Medium VRAM - 8-12GB)
```bash
python dgls_train.py --config config.toml \
    --dynamic_swapping \
    --initial_gpu_layers 5 \
    --final_gpu_layers 5 \
    --prefetch 2 \
    --batch_move \
    --threading \
    --selective_packing 64\
    --event_sync
```

### Maximum Performance (High VRAM - 16GB+)
```bash
python dgls_train.py --config config.toml \
    --dynamic_swapping \
    --gpu_layers "0,1,2,3,18,19,20,21,22" \
    --prefetch 4 \
    --threading \
    --cuda_streams \
    --batch_move \
    --selective_packing 128 \
    --async_zero_grad \
    --compile \
    --event_sync
```

### Mixed Precision Optimization
```bash
python dgls_train.py --config config.toml \
    --dynamic_swapping \
    --cast_target f32 bf16 \
    --mixed_precision bf16 \
    --initial_gpu_layers 2 \
    --prefetch 2 \
    --selective_packing 128
```

### Environment Setup
```bash
# Set these if experiencing startup issues
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0 # if you have more than 1 gpu

# Run training
python dgls_train.py --config your_config.toml [your_args]
```

## Limitations & Compatibility

**Single GPU Only**: Designed for world_size=1 training. While you can specify different GPUs if you have multiple cards, each training session uses only one GPU at a time.

**Optimizer Compatibility Warnings**:
- **PROCEED WITH CAUTION**: ZeRO optimizers, sharded optimizers, gradient release, and GenericOptim may interfere with the swapping algorithm
- **Recommended**: Use standard optimizers
- **Test thoroughly** if using advanced optimizer configurations

**Model Architecture Notes**:
- Mixed layer type models (e.g., SDXL) may require specific layer targeting with `--gpu_layers`
- Some layers don't swap well - use precise layer control when needed
- Test with your specific model architecture before production training

**Performance Limitations**:
- RAM speed directly impacts swapping performance (3600MHz+ recommended, as this is the lowest I've tested)
- PCIe bandwidth affects transfer speeds, check your card model speeds
- Threading may cause instability on some systems

## Troubleshooting

### Common Issues

**"CUDA out of memory" during startup**:
```bash
# Solution: Reduce permanent GPU layers
--initial_gpu_layers 1 --final_gpu_layers 1
# Or even more conservative:
--initial_gpu_layers 0 --final_gpu_layers 1
```

**Slow training speed**:
```bash
# Solution: Enable optimizations
--threading --prefetch 1
# For larger GPUs, also add:
--cuda_streams --prefetch 2
```

**Low prefetch hit rate (<80%)**:
- Reduce `--prefetch` value
- Enable `--threading` for background management
- Check for memory pressure causing evictions
- Monitor GPU memory usage and reduce resident layers- Task manager is good to have open for RAM and VRAM

**Training crashes with threading**:
```bash
# Solution: Disable threading and use conservative settings
--prefetch 1  # Remove --threading flag
--device_sync_mode late
```

**Batch move causing device issues**:
```bash
# Solution: Disable batch operations
# Remove --batch_move flag, use individual layer transfers
```

### Environment Setup Issues

**Startup problems**:
```bash
# Set these environment variables manually
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export CUDA_LAUNCH_BLOCKING=1  # For debugging only
```

**DeepSpeed initialization errors**:
- Ensure single GPU setup (world_size=1)
- Check CUDA compatibility with PyTorch version
- Verify DeepSpeed installation

### Performance Debugging

**Monitor these metrics**:
- **Prefetch Hit Rate**: Should be >85%
- **GPU Memory Usage**: Keep under 90% to avoid OOM
- **Transfer Speeds**: Check if RAM speed is bottleneck
- **Step Duration**: Compare with/without swapping, threading, prefetch etc

**Verbose debugging**:
```bash
--verbose  # Enable detailed logging and timing information
```

###Notes
- Maintain hot path optimizations (avoid abstraction overhead)
- This was only tested on 2060 (6GB) and 2080ti (11GB)

## Attribution & License

**Dynamic Swapping Method**: Original research and implementation by **obisin**
**Core Innovation**: Smart dynamic layer swapping between GPU and CPU with predictive prefetching, copy-compute overlap, and mixed precision awareness.
**Community**: Thanks to all testers, contributors, and users providing feedback

### License

**MIT License** - See LICENSE file for full details

```
MIT License

Copyright (c) 2025 obisin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


**Usage Compliance**: Users are responsible for compliance with licenses of any underlying training frameworks they integrate with DGLS.
