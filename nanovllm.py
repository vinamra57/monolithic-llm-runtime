"""
Monolithic LLM Runtime - High-performance inference engine
All components integrated into a single module for maximum performance
Target: 2000+ tok/s throughput
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import List, Optional, Union, Dict
from dataclasses import dataclass
import time
from collections import defaultdict
import os

# Enable memory optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


@dataclass
class SamplingParams:
    """Sampling parameters for text generation"""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 16
    ignore_eos: bool = False

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


class KVCache:
    """Monolithic KV cache for attention - no separate module"""
    def __init__(self, num_layers: int, batch_size: int, num_heads: int, head_dim: int,
                 max_seq_len: int, device: str, dtype: torch.dtype):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Preallocate cache tensors for all layers
        self.k_cache = torch.zeros(
            (num_layers, batch_size, num_heads, max_seq_len, head_dim),
            dtype=dtype, device=device
        )
        self.v_cache = torch.zeros(
            (num_layers, batch_size, num_heads, max_seq_len, head_dim),
            dtype=dtype, device=device
        )
        self.seq_lens = torch.zeros(batch_size, dtype=torch.long, device=device)

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int):
        """Update cache for a specific layer"""
        batch_size, num_heads, seq_len, head_dim = k.shape
        end_pos = start_pos + seq_len

        self.k_cache[layer_idx, :batch_size, :, start_pos:end_pos, :] = k
        self.v_cache[layer_idx, :batch_size, :, start_pos:end_pos, :] = v

        return (
            self.k_cache[layer_idx, :batch_size, :, :end_pos, :],
            self.v_cache[layer_idx, :batch_size, :, :end_pos, :]
        )

    def reset(self):
        """Reset cache for new batch"""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_lens.zero_()


class LLM:
    """
    Monolithic LLM Runtime
    All inference logic integrated for maximum performance
    """

    def __init__(
        self,
        model_path: str,
        enforce_eager: bool = True,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.device = device
        self.enforce_eager = enforce_eager

        print(f"Loading model from {model_path}...")

        # Load model configuration
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # Determine dtype
        if dtype == "auto":
            self.dtype = torch.float16 if device == "cuda" else torch.float32
        else:
            self.dtype = getattr(torch, dtype)

        # Load model (SDPA attention is used by default, which automatically
        # uses Flash Attention if available)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            device_map=device,
            trust_remote_code=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set model to eval mode
        self.model.eval()

        # Compile model for faster execution (if not in eager mode)
        if not enforce_eager and device == "cuda":
            try:
                print("  Compiling model with torch.compile()...")
                # Use "default" mode to avoid CUDA graph issues with KV cache
                self.model = torch.compile(self.model, mode="default")
                print("  Model compiled successfully!")
            except Exception as e:
                print(f"  torch.compile() failed: {e}, using uncompiled model")

        # Extract model properties
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.vocab_size = self.config.vocab_size

        # Enable optimizations
        if not enforce_eager and device == "cuda":
            self._enable_cuda_optimizations()

        print(f"Model loaded: {self.config.model_type}")
        print(f"  Layers: {self.num_layers}, Heads: {self.num_heads}, Hidden: {self.config.hidden_size}")
        print(f"  Vocab size: {self.vocab_size}, Max length: {max_model_len}")
        print(f"  Device: {device}, Dtype: {self.dtype}")

    def _enable_cuda_optimizations(self):
        """Enable CUDA-specific optimizations"""
        try:
            # Try to use Flash Attention if available
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("  CUDA optimizations enabled: Flash Attention, Memory Efficient Attention")
        except:
            print("  Flash Attention not available, using standard attention")

        # Enable TF32 for faster computation on Ampere GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _sample_tokens(
        self,
        logits: torch.Tensor,
        sampling_params: List[SamplingParams],
    ) -> torch.Tensor:
        """
        Sample next tokens from logits - monolithic sampling logic

        Args:
            logits: [batch_size, vocab_size]
            sampling_params: list of sampling parameters for each sequence

        Returns:
            next_tokens: [batch_size]
        """
        batch_size = logits.shape[0]
        next_tokens = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i in range(batch_size):
            params = sampling_params[i]
            logit = logits[i]

            # Apply temperature
            if params.temperature > 0:
                logit = logit / params.temperature

            # Apply top-k filtering
            if params.top_k > 0:
                indices_to_remove = logit < torch.topk(logit, params.top_k)[0][..., -1, None]
                logit[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if params.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logit, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > params.top_p
                # Keep at least one token
                sorted_indices_to_remove[0] = False

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logit[indices_to_remove] = float('-inf')

            # Sample from the filtered distribution
            if params.temperature == 0:
                next_token = torch.argmax(logit)
            else:
                probs = F.softmax(logit, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze()

            next_tokens[i] = next_token

        return next_tokens

    @torch.inference_mode()
    def generate(
        self,
        prompts: Union[List[str], List[List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]],
        use_tqdm: bool = True,
    ) -> List[str]:
        """
        Generate text for batches with optimized memory-efficient batching

        Args:
            prompts: list of prompt strings or token ID lists
            sampling_params: single SamplingParams or list of SamplingParams per prompt
            use_tqdm: whether to show progress bar

        Returns:
            list of generated texts
        """
        # Handle single sampling params
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)

        # Convert to token IDs if needed
        if isinstance(prompts[0], str):
            prompt_token_lists = []
            for p in prompts:
                tokens = self.tokenizer(p, return_tensors="pt").input_ids[0].tolist()
                prompt_token_lists.append(tokens)
        else:
            prompt_token_lists = prompts

        num_requests = len(prompt_token_lists)

        # Sort by prompt length to optimize batching
        sorted_indices = sorted(range(num_requests), key=lambda i: len(prompt_token_lists[i]))

        # Process in batches to avoid OOM
        # Trying batch size 60 to push higher
        PREFILL_BATCH_SIZE = 60
        DECODE_BATCH_SIZE = 64

        all_outputs = [[] for _ in range(num_requests)]

        # Process batches
        for batch_start in range(0, num_requests, PREFILL_BATCH_SIZE):
            batch_end = min(batch_start + PREFILL_BATCH_SIZE, num_requests)
            batch_indices = sorted_indices[batch_start:batch_end]

            # Prepare batch
            batch_prompts = [prompt_token_lists[i] for i in batch_indices]
            batch_params = [sampling_params[i] for i in batch_indices]

            # Pad prompts
            max_prompt_len = max(len(p) for p in batch_prompts)
            input_ids = torch.zeros((len(batch_prompts), max_prompt_len),
                                   dtype=torch.long, device=self.device)
            input_ids.fill_(self.tokenizer.pad_token_id or 0)

            for i, prompt in enumerate(batch_prompts):
                input_ids[i, :len(prompt)] = torch.tensor(prompt, dtype=torch.long)

            # Prefill phase - process all prompts
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

            # Sample first tokens
            next_tokens = self._sample_tokens(logits, batch_params)

            # Initialize generation state
            max_new_tokens = max(p.max_tokens for p in batch_params)
            # Pre-allocate output buffer to avoid torch.cat overhead
            batch_output_ids = torch.zeros(
                (len(batch_prompts), max_new_tokens),
                dtype=torch.long, device=self.device
            )
            batch_output_ids[:, 0] = next_tokens

            finished = torch.zeros(len(batch_prompts), dtype=torch.bool, device=self.device)
            generated_counts = torch.ones(len(batch_prompts), dtype=torch.long, device=self.device)

            # Decode phase - generate tokens one by one
            for step in range(1, max_new_tokens):
                if finished.all():
                    break

                # Forward pass with last generated tokens
                outputs = self.model(
                    input_ids=next_tokens.unsqueeze(1),
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

                # Sample next tokens
                next_tokens = self._sample_tokens(logits, batch_params)

                # Update outputs (in-place, no allocation)
                batch_output_ids[:, step] = next_tokens
                generated_counts += ~finished

                # Check for completion
                for i in range(len(batch_prompts)):
                    if finished[i]:
                        continue

                    if generated_counts[i] >= batch_params[i].max_tokens:
                        finished[i] = True
                    elif not batch_params[i].ignore_eos and next_tokens[i] == self.tokenizer.eos_token_id:
                        finished[i] = True

            # Store outputs (trim to actual generated length)
            for i, idx in enumerate(batch_indices):
                actual_len = generated_counts[i].item()
                all_outputs[idx] = batch_output_ids[i, :actual_len].cpu().tolist()

        # Decode all outputs
        generated_texts = []
        for output_ids in all_outputs:
            if output_ids:
                text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            else:
                text = ""
            generated_texts.append(text)

        return generated_texts


def main():
    """Quick test of the monolithic runtime"""
    print("Monolithic LLM Runtime Test")
    print("=" * 50)

    # This is just a placeholder - actual model path will be on the VM
    # For now, we'll just show that the code structure is correct
    print("Runtime implementation complete!")
    print("Ready for deployment to VM and benchmarking")


if __name__ == "__main__":
    main()
