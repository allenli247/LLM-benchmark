import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Baseline LLM Inference Runner")
    parser.add_argument("--model", type=str, default="google/gemma-2b", help="HuggingFace model ID")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Number of tokens to generate")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {args.model} on {device}...")
    
    # NOTE: You may need to have run `huggingface-cli login` and/or accept the Gemma license on HF
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("If using Gemma, ensure you have accepted the conditions on the Hugging Face website and are authenticated.")
        return

    # Basic benchmarking logic
    print("Preparing input...")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    
    print("Warming up GPU cache...")
    _ = model.generate(**inputs, max_new_tokens=10)
    if device == "cuda":
        torch.cuda.synchronize()

    print(f"\nRunning benchmark for '{args.model}'...")
    print("-" * 50)
    
    # 1. Time to First Token (TTFT)
    # We measure it by generating exactly 1 token (which includes prefill + 1 decode)
    if device == "cuda":
        torch.cuda.synchronize()
    ttft_start = time.time()
    _ = model.generate(**inputs, max_new_tokens=1)
    if device == "cuda":
        torch.cuda.synchronize()
    ttft_end = time.time()
    
    ttft = ttft_end - ttft_start
    print(f"Time to First Token (TTFT):   {ttft * 1000:.2f} ms")
    
    # 2. Tokens per Second (TPS)
    if device == "cuda":
        torch.cuda.synchronize()
    tps_start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    if device == "cuda":
        torch.cuda.synchronize()
    tps_end = time.time()
    
    total_time = tps_end - tps_start
    # Calculate how many new tokens were actually generated
    prompt_length = inputs["input_ids"].shape[1]
    num_generated_tokens = outputs.shape[1] - prompt_length
    
    # Technically TPS during pure decode is tokens / (total_time - TTFT), 
    # but average TPS is total_tokens / total_time. We'll show average TPS.
    tps = num_generated_tokens / total_time
    
    print(f"Generated {num_generated_tokens} tokens in {total_time:.2f} s")
    print(f"Average Tokens per Second:    {tps:.2f} tok/s")
    print("-" * 50)
    
    # Decode final output for sanity check
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nSample Output:")
    print(generated_text)

if __name__ == "__main__":
    main()
