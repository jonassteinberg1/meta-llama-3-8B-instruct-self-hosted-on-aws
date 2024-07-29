from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
import torch

# Define paths
tokenizer_path = 'meta-llama/Meta-Llama-3-8B-Instruct/original'

# Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

# Load the model with automatic device placement and half-precision
model = LlamaForCausalLM.from_pretrained(tokenizer_path, device_map='auto', torch_dtype=torch.float16)

# Ensure model is in evaluation mode
model.eval()

# Define the function to generate a response
def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Start the interactive loop with an initial prompt

print("Enter 'quit' to stop the conversation.")
while True:
    user_input = input()
    if user_input.lower() == 'quit':
        break
    context = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are Llama, an advanced AI designed to answer questions and provide information succinctly on a wide range of topics. Your responses should be clear, concise, and directly address the user's queries. Avoid answering in weird formats like multiple choice or lists unless explicitly asked. Maintain a serious and practical tone in your answers. Do not use sarcasm, humor, cuteness, or irreverence in your responses.<|eot_id|><|start_header_id|>user<|end_header_id|>{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    response = generate_response(context)
    print(f"Llama: {response}")
