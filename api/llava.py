from PIL import Image
import torch
import transformers
from transformers import pipeline
import pdb
import warnings
warnings.filterwarnings("ignore")

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)
def call_llava(text_query, image_path, model, temperature=0.0, extra_explanation=True):
    """
    Args:
        text_query: str. The query text.
        image_path: str. The path to the image.
        temperature: float. The temperature for the generation.
    """
    if model=='llava-7b':
        llava_pipeline = pipeline(
            model = "llava-hf/llava-1.5-7b-hf",
            device_map="auto",
        )
    elif model=='llava-13b':
        llava_pipeline = pipeline(
            model = "llava-hf/llava-1.5-13b-hf",
            device_map="auto",
        )
    elif model=='llava-34b':
        llava_pipeline = pipeline(
            model = "liuhaotian/llava-v1.6-34b",
            device_map="auto",
        )
    elif model=='llava-sci':
        llava_pipeline = pipeline(
            model = "liuhaotian/llava-lcs558k-scienceqa-vicuna-13b-v1.3",
            device_map="auto",
        )
    elif model=='llava-mistral':
        llava_pipeline = pipeline(
            model = "llava-hf/llava-v1.6-mistral-7b-hf",
            device_map="auto",
        )        

    img = Image.open(image_path)
    prompt = "<image>\nUSER: " + text_query + "\nASSISTANT:"
    response = llava_pipeline(prompt=prompt, images=img, max_new_tokens=1024)[0]['generated_text']
    if extra_explanation:
        bool_ans = response.split("\nASSISTANT:")[1].lstrip()
        # Getting explanations
        prompt = prompt + bool_ans + "\nUSER: Provide your detailed explanations. \nASSISTANT:"
        explanation = llava_pipeline(prompt=prompt, images=img, max_new_tokens=1024)[0]['generated_text']
        explanation = explanation.split("\nASSISTANT:")[-1].lstrip()
        response = bool_ans + '\n' + explanation
    else:
        response = response.split("\nASSISTANT:")[1].lstrip()
    return response

if __name__ == "__main__":
    text_query = "What is in the image?"
    image_path = "../images/mol_caption_0.png"
    # pdb.set_trace()
    print(call_llava(text_query, image_path))