import torch

from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

# Define paths
tokenizer_path = 'meta-llama/Meta-Llama-3-8B-Instruct/original'

# Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

# Load the model with automatic device placement and half-precision
model = LlamaForCausalLM.from_pretrained(tokenizer_path, device_map='auto', torch_dtype=torch.float16)

# Ensure model is in evaluation mode
model.eval()

#def generate_text(prompt):
#    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
#    with torch.no_grad():
#        outputs = model.generate(inputs.input_ids, max_length=528, num_return_sequences=1)
#    return tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#print("Enter your prompt:")
#
#while True:
#    try:
#        # Read from standard input
#        prompt = input()
#
#        # Generate the output using the model
#        output = generate_text(prompt)
#
#        # Print the output to standard output
#        print(output)
#    except EOFError:
#        break

# Define the function to generate a response
def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Start the interactive loop
print("Enter 'quit' to stop the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = generate_response(user_input)
    print(f"Llama: {response}")
