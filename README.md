# Monolithic LLM Runtime

High-performance LLM inference runtime with monolithic architecture targeting 2000+ tok/s throughput.

## Architecture

All components integrated into a single `nanovllm.py` module:
- Model loading (HuggingFace transformers)
- KV cache management
- Token generation with sampling
- Batch processing
- CUDA optimizations

## Setup

```bash
pip install -r requirements.txt
```

## Running Benchmarks

```bash
python bench.py
```

The benchmark expects the model at `~/huggingface/Qwen3-0.6B/`.

## Performance Target

- **Goal**: 2000 tok/s throughput
- **Test**: 256 sequences, random input/output lengths up to 1024 tokens
- **Model**: Qwen3-0.6B

## Key Optimizations

1. Integrated KV cache (no separate module overhead)
2. Batch processing with past_key_values reuse
3. CUDA optimizations (Flash Attention, TF32)
4. Minimal abstraction layers
5. Direct tensor operations

## VM Deployment

The code is designed to run on the GCP VM (researchvm-ubuntu) which has all dependencies pre-installed.
