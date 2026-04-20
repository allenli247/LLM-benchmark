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
    torch_dtype=torch.bfloat16,
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
# 6. Generate the output
outputs = model.generate(
    **inputs,
    max_new_tokens=100, # Limit the output length
    do_sample=True,     # Allow for some creativity
    temperature=0.7     # Control the randomness
)

# 7. Decode the tokens back into readable text
# We slice [inputs['input_ids'].shape[1]:] to remove the prompt from the final output string
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("\n--- Output ---")
print(response)


