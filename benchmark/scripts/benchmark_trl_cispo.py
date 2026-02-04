#!/usr/bin/env python
"""
Benchmark script to compare TRL GRPOTrainer CISPO loss: standard vs Liger Kernel.

This script measures the training speed for CISPO loss type with:
1. Standard TRL implementation
2. Liger Kernel loss only (no other kernel replacements like RMSNorm, SwiGLU)

Usage:
    python benchmark/scripts/benchmark_trl_cispo.py
"""

import gc
import time

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B"  # Larger model for realistic benchmarking
NUM_TRAIN_STEPS = 30
BATCH_SIZE = 4  # Must be >= num_generations
MAX_PROMPT_LENGTH = 1024
MAX_COMPLETION_LENGTH = 1024
NUM_GENERATIONS = 4


def create_dummy_dataset(num_samples: int = 100) -> Dataset:
    """Create a dummy dataset for benchmarking."""
    data = {
        "prompt": [
            f"What is {i} + {i}? Answer: "
            for i in range(num_samples)
        ],
    }
    return Dataset.from_dict(data)


def dummy_reward_fn(completions, prompts=None, **kwargs):
    """Dummy reward function that returns random rewards."""
    return [torch.rand(1).item() for _ in completions]


class LigerLossOnlyGRPOTrainer(GRPOTrainer):
    """
    Custom GRPOTrainer that uses Liger loss function only,
    without applying other Liger kernels (RMSNorm, SwiGLU, etc.)
    """
    
    def __init__(self, *args, **kwargs):
        # Force use_liger_kernel=False to avoid applying other kernels
        if 'args' in kwargs and kwargs['args'] is not None:
            kwargs['args'].use_liger_kernel = False
        super().__init__(*args, **kwargs)
        
        # Initialize Liger GRPO loss manually
        self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
            beta=self.beta,
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
            temperature=self.temperature,
            use_ref_model=self.beta != 0.0,
            loss_type=self.loss_type,
            max_completion_length=self.max_completion_length,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        # Use Liger loss
        unwrapped_model = self.accelerator.unwrap_model(model)
        return self._compute_liger_loss(unwrapped_model, inputs)
    
    def _compute_liger_loss(self, model, inputs):
        """Compute loss using Liger's fused linear GRPO loss."""
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        # Get last hidden state (similar to TRL's approach)
        logits_to_keep = completion_ids.size(1)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            logits_to_keep=logits_to_keep + 1,
        )
        last_hidden_state = outputs.hidden_states[-1]
        
        # Only keep completion part - take the last logits_to_keep tokens
        last_hidden_state = last_hidden_state[:, -logits_to_keep - 1 : -1, :]
        
        # Ensure dtype matches lm_head weight
        if last_hidden_state.dtype != model.lm_head.weight.dtype:
            last_hidden_state = last_hidden_state.to(model.lm_head.weight.dtype)
        
        # Compute loss using Liger
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"],
            bias=model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
        )
        
        # Log metrics
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]
        
        mode = "train" if self.model.training else "eval"
        if mean_kl is not None:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())
        
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
        return loss / normalizer


def run_benchmark(
    mode: str = "standard",  # "standard", "liger_loss_only", "liger_full"
    loss_type: str = "cispo",
    num_steps: int = NUM_TRAIN_STEPS,
) -> dict:
    """
    Run a single benchmark with the specified configuration.
    
    Args:
        mode: Benchmark mode - "standard", "liger_loss_only", or "liger_full"
        loss_type: Loss type (e.g., "cispo", "dapo", "grpo")
        num_steps: Number of training steps
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Running benchmark: mode={mode}, loss_type={loss_type}")
    print(f"{'='*60}")
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # Use SDPA instead of flash_attention_2
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    dataset = create_dummy_dataset(num_samples=num_steps * BATCH_SIZE * 2)
    
    # Configure trainer
    epsilon_high = 5.0 if loss_type == "cispo" else 0.2
    
    # Set use_liger_kernel based on mode
    use_liger_kernel = (mode == "liger_full")
    
    config = GRPOConfig(
        output_dir=f"./tmp_benchmark_{mode}_{loss_type}",
        num_train_epochs=1,
        max_steps=num_steps,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        # GRPO specific
        loss_type=loss_type,
        epsilon=0.2,
        epsilon_high=epsilon_high,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=NUM_GENERATIONS,
        # Liger kernel for full mode
        use_liger_kernel=use_liger_kernel,
        bf16=True,
        bf16_full_eval=True,
    )
    
    # Choose trainer class based on mode
    if mode == "liger_loss_only":
        trainer = LigerLossOnlyGRPOTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            reward_funcs=dummy_reward_fn,
        )
    else:
        # "standard" or "liger_full" - use standard GRPOTrainer
        trainer = GRPOTrainer(
            model=model,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            reward_funcs=dummy_reward_fn,
        )
    
    # Clear memory before training
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print("Starting training...")
    
    # Benchmark training
    print(f"Starting training benchmark for {num_steps} steps...")
    start_time = time.perf_counter()
    start_mem = torch.cuda.memory_allocated()
    
    trainer.train()
    
    end_time = time.perf_counter()
    peak_mem = torch.cuda.max_memory_allocated()
    
    total_time = end_time - start_time
    time_per_step = total_time / num_steps
    
    results = {
        "mode": mode,
        "loss_type": loss_type,
        "num_steps": num_steps,
        "total_time_s": total_time,
        "time_per_step_s": time_per_step,
        "peak_memory_gb": peak_mem / (1024**3),
        "start_memory_gb": start_mem / (1024**3),
    }
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Time per step: {time_per_step:.4f}s")
    print(f"  Peak memory: {peak_mem / (1024**3):.2f} GB")
    
    # Cleanup
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


def main():
    """Run the full benchmark comparison."""
    print("="*70)
    print("TRL CISPO Benchmark: Standard vs Liger Loss Only vs Liger Full")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max completion length: {MAX_COMPLETION_LENGTH}")
    print(f"  Num generations: {NUM_GENERATIONS}")
    print(f"  Num training steps: {NUM_TRAIN_STEPS}")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("\nERROR: CUDA is not available. This benchmark requires a GPU.")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
    results = {}
    
    # 1. Benchmark CISPO with standard TRL (no Liger)
    try:
        results["standard"] = run_benchmark(mode="standard", loss_type="cispo")
    except Exception as e:
        print(f"Error running standard benchmark: {e}")
        import traceback
        traceback.print_exc()
        results["standard"] = None
    
    # 2. Benchmark CISPO with Liger Loss only (no other kernels)
    try:
        results["liger_loss_only"] = run_benchmark(mode="liger_loss_only", loss_type="cispo")
    except Exception as e:
        print(f"Error running Liger Loss Only benchmark: {e}")
        import traceback
        traceback.print_exc()
        results["liger_loss_only"] = None
    
    # 3. Benchmark CISPO with Liger Full (all kernels: RMSNorm, SwiGLU, Loss, etc.)
    try:
        results["liger_full"] = run_benchmark(mode="liger_full", loss_type="cispo")
    except Exception as e:
        print(f"Error running Liger Full benchmark: {e}")
        import traceback
        traceback.print_exc()
        results["liger_full"] = None
    
    # Print comparison
    print("\n" + "="*90)
    print("BENCHMARK COMPARISON")
    print("="*90)
    
    r_std = results.get("standard")
    r_loss = results.get("liger_loss_only")
    r_full = results.get("liger_full")
    
    if r_std:
        print(f"\n{'Metric':<25} {'TRL Standard':<18} {'Liger Loss Only':<18} {'Liger Full':<18} {'Loss vs Std':<15} {'Full vs Std':<15}")
        print("-"*107)
        
        std_time = r_std["time_per_step_s"]
        std_mem = r_std["peak_memory_gb"]
        
        loss_time = r_loss["time_per_step_s"] if r_loss else float('nan')
        loss_mem = r_loss["peak_memory_gb"] if r_loss else float('nan')
        
        full_time = r_full["time_per_step_s"] if r_full else float('nan')
        full_mem = r_full["peak_memory_gb"] if r_full else float('nan')
        
        # Time comparison
        loss_speedup = std_time / loss_time if r_loss else float('nan')
        full_speedup = std_time / full_time if r_full else float('nan')
        print(f"{'Time per step (s)':<25} {std_time:<18.4f} {loss_time:<18.4f} {full_time:<18.4f} {loss_speedup:.2f}x{'↑' if loss_speedup > 1 else '↓':<10} {full_speedup:.2f}x{'↑' if full_speedup > 1 else '↓'}")
        
        # Memory comparison
        loss_mem_red = (1 - loss_mem / std_mem) * 100 if r_loss else float('nan')
        full_mem_red = (1 - full_mem / std_mem) * 100 if r_full else float('nan')
        print(f"{'Peak memory (GB)':<25} {std_mem:<18.2f} {loss_mem:<18.2f} {full_mem:<18.2f} {abs(loss_mem_red):.1f}%{'↓' if loss_mem_red > 0 else '↑':<10} {abs(full_mem_red):.1f}%{'↓' if full_mem_red > 0 else '↑'}")
        
        # Total time
        std_total = r_std["total_time_s"]
        loss_total = r_loss["total_time_s"] if r_loss else float('nan')
        full_total = r_full["total_time_s"] if r_full else float('nan')
        print(f"{'Total time (s)':<25} {std_total:<18.2f} {loss_total:<18.2f} {full_total:<18.2f}")
    else:
        print("\nCould not complete comparison - standard benchmark failed.")
        for mode, r in results.items():
            if r:
                print(f"\n{mode}:")
                print(f"  Time per step: {r['time_per_step_s']:.4f}s")
                print(f"  Peak memory: {r['peak_memory_gb']:.2f} GB")


if __name__ == "__main__":
    main()
