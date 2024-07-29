import runhouse as rh
import torch

class HFChatModel(rh.Module):
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", **model_kwargs):
        super().__init__()
        self.model_id, self.model_kwargs = model_id, model_kwargs
        self.pipeline = None

    def load_model(self):
        import transformers

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs=self.model_kwargs,
            device="cuda",
        )

    def predict(self, prompt_text, **inf_kwargs):
        if not self.pipeline:
            self.load_model()

        messages = [
            {
                "role": "system",
                "content": "You are an ai designed to answer questions about technology",
            },
            {"role": "user", "content": prompt_text},
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        return outputs[0]["generated_text"][len(prompt) :]

if __name__ == "__main__":

    # Create a cluster with the desired instance type and provider
    gpu = rh.cluster(
        name="rh-a10x", instance_type="A10G:1", memory="32+", provider="aws", 
        ssh_creds={"ssh_user": "ubuntu", "ssh_private_key": "/root/.ssh/sky-key"}, region="us-west-2",
    ).up_if_not()

    # Define the environment for our module
    env = rh.env(
        reqs=[
            "torch",
            "transformers",
            "accelerate",
            "bitsandbytes",
            "safetensors",
            "scipy",
        ],
        secrets=["huggingface"],  # Needed to download Llama 3 from Hugging Face
        name="llama3inference",
        working_dir="./",
    )

    # Define our module and run it on the remote cluster
    remote_hf_chat_model = HFChatModel(
        torch_dtype=torch.bfloat16,
    ).get_or_to(gpu, env=env, name="llama3-8b-model")

    # Calling our remote function
    while True:
        prompt = input(
            "\n\n... Enter a prompt to chat with the model, and 'exit' to exit ...\n"
        )
        if prompt.lower().strip() == "exit":
            break
        output = remote_hf_chat_model.predict(prompt)
        print("\n\n... Model Output ...\n")
        print(output)
