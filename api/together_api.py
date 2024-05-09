import together
from time import sleep

together.api_key = "770015d79c016679c75071f69d9c72b7f892285d646298d4775e743be18bbec5"

MODEL_CKPTS = {
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "llama2_70b": "meta-llama/Llama-2-70b-chat-hf",
    "llama2_13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama2_7b": "meta-llama/Llama-2-7b-chat-hf",
}

MIXTRAL_PROMPT = "[INST] {text_query} [/INST]"

LLAMA_PROMPT = """[INST] <<SYS>>
{system_content}
<</SYS>>

{text_query} [/INST]"""

PROMPT_TEMPLATE = {
    "mixtral": MIXTRAL_PROMPT,
    "llama2_70b": LLAMA_PROMPT,
    "llama2_13b": LLAMA_PROMPT,
    "llama2_7b": LLAMA_PROMPT,
}

STOP_TOKENS = ["</s>", "<human>"]

def prepare_input(
    text_query: str, system_content: str, model: str = "mixtral"
) -> str:
    input_text = PROMPT_TEMPLATE[model].format(
        system_content=system_content, text_query=text_query
    )
    return input_text


def call_together(
    text_query: str, system_content: str, model: str = "mixtral", temperature: float = 0.0
):
    prompt = prepare_input(text_query, system_content, model)
    # Small hack: LLaMA does not support max_tokens = 4096 for some reason...
    max_tokens = 4096 if model == "mixtral" else 2048
    output = together.Complete.create(
        prompt=prompt,
        model=MODEL_CKPTS[model],
        # model="meta-llama/Llama-2-70b-chat-hf",
        max_tokens=max_tokens,
        temperature=temperature,
        stop=STOP_TOKENS,
    )
    # sleep
    sleep(1)

    

    generatedText = output['output']['choices'][0]['text']
    return generatedText