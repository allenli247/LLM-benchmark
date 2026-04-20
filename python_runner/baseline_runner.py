import time  # <-- Added this import!
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Define the model ID
model_id = "google/gemma-4-E2B-it"

print("Loading tokenizer and model...")
# 2. Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Load the Model directly onto the GPU
# We use bfloat16 (brain floating point) to save VRAM while maintaining accuracy
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda", 
    dtype=torch.bfloat16,
)

# 4. Prepare your prompt
prompt = "Explain the difference between Python and C++ in two sentences."

# Gemma-IT expects a specific chat template formatting
chat = [
    { "role": "user", "content": prompt },
]
formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

# 5. Tokenize the input and move it to the GPU
inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

print("Generating response...")

# 1. Force the CPU to wait until the GPU is fully ready
torch.cuda.synchronize()

# 2. Start the timer
start_time = time.perf_counter()

# 3. Generate the output
outputs = model.generate(
    **inputs,
    max_new_tokens=100, 
    do_sample=True,     
    temperature=0.7     
)

# 4. Force the CPU to wait until the GPU has completely finished generating
torch.cuda.synchronize()

# 5. Stop the timer
end_time = time.perf_counter()

# --- Calculate the Metrics --- (Removed the duplicate block)
total_time = end_time - start_time

# The output includes both the prompt tokens AND the generated tokens.
# We need to subtract the prompt length to find out how many new tokens were made.
input_length = inputs['input_ids'].shape[1]
total_output_length = outputs.shape[1]
generated_tokens = total_output_length - input_length

# Calculate Tokens Per Second (TPS)
tps = generated_tokens / total_time

# Decode and print the response
response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

# --- Format the output ---
# Using an f-string to create a clean, formatted block of text
output_text = f"""
--- Benchmark Run ---
Prompt: {prompt}

Model Response:
{response}

--- Baseline Metrics ---
Tokens Generated: {generated_tokens}
Total Time:       {total_time:.4f} seconds
Tokens Per Sec:   {tps:.2f} TPS
-----------------------
"""

# Print to the console so you can still watch it happen live
print(output_text)

# --- Save to a text file ---
# Open a file named 'benchmark_results.txt' in append mode ('a')
with open("benchmark_results.txt", "a", encoding="utf-8") as f:
    f.write(output_text + "\n")

print("Results successfully saved to benchmark_results.txt")