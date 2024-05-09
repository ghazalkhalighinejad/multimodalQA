import torch
import transformers
from transformers import AutoTokenizer
import pdb
import os 
from huggingface_hub import HfApi, HfFolder
import warnings
warnings.filterwarnings("ignore")

# set api for login and save token
token = os.environ['HUGGINGFACE_TOKEN']
model = "meta-llama/Llama-2-70b-chat-hf"
transformers.utils.logging.set_verbosity(transformers.logging.ERROR)
tokenizer = AutoTokenizer.from_pretrained(model, token=token)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def call_llama2_70b(text_query, system_content = "", temperature=0.0):
    input_text = f"""<s>[INST] <<SYS>>
{system_content}
<</SYS>>

{text_query} [/INST]"""
    if temperature == 0.0:
        sequences = pipeline(
            input_text,
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4096,
            temperature=temperature,
            top_k=None, top_p=None
        )
    else:
        sequences = pipeline(
            input_text,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=4096,
        )

    response = sequences[0]['generated_text'] 
    return response.removeprefix(input_text).lstrip()


if __name__ == "__main__":
    print(call_llama2_70b("Where is Los Angeles?"))