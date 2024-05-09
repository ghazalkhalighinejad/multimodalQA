import os
import torch
import numpy as np 
import pandas as pd
import json 
from tqdm import tqdm
import argparse

from api.gpt4 import (
    call_gpt4_vision, 
    call_gpt4, 
    call_gpt3_5
)

from api.gemini import (
    call_gemini_pro_vision, 
    call_gemini_pro,
    call_palm_2
)

from api.claude3 import (
    call_claude3_vision,
    call_claude3
)
from api.together_api import call_together


def call_vision(text_query, image_path, model):
    if model == "gemini":
        response = call_gemini_pro_vision(text_query, image_path)
    elif model == "gpt4":
        response = call_gpt4_vision(text_query, image_path)
    elif model == "llava":
        response = call_llava(text_query, image_path)
    elif model == "claude3":
        response = call_claude3_vision(text_query, image_path)
    else:
        raise NotImplementedError
    return response

def call_text(text_query, system_content, model):
    if model == "gemini":
        response = call_gemini_pro(f"{system_content}\n{text_query}")
    elif model == "palm2":
        response = call_palm_2(f"{system_content}\n{text_query}")
    elif model == "gpt4":
        response = call_gpt4(text_query, system_content)
    elif model == "gpt3.5":
        response = call_gpt3_5(text_query, system_content)
    elif model == "claude3":
        response = call_claude3(text_query, system_content)
    elif ("llama" in args.model or args.model == "mixtral") and args.together:
        response = call_together(text_query, system_content, model=args.model)
    else:
        raise NotImplementedError
    return response


def prepare_vision_prompt(dataset):

    prompts = []

    for row in dataset:
        
        id, question, choices, label, description = row["id"], row["question"], row["choices"], row["label"], row["description"]

        prompt = f"""Given the image of a chemical compound, answer the following multiple choice question.

        Question: {question}
        Choices: 
        1. {choices[0]}
        2. {choices[1]}
        3. {choices[2]}
        4. {choices[3]}

        Your answer should be a number between 1 and 4.
        """
        id = row["id"].split("/")[-1]
        image = f"images/{id}.png"

        prompts.append({
            "id": id,
            "text_query": prompt,
            "image_path": image,
            "true_label": label
        })

    return prompts


def parse_answer(response):

    response = ''.join(filter(str.isdigit, response))
    response = int(response)

    return response



def check_correctness(response, true_label):

    # if response == true_label:
    return response == true_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gemini", help="model name to evaluate")
    parser.add_argument('--together', action="store_true", help="use together api")
    parser.add_argument('--resume', action="store_true", help="resume from last checkpoint")
    args = parser.parse_args()

    if args.model == "llava":
        from api.llava import call_llava
   
    result_dir = "chemistry_benchmark"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    result_path = f"{result_dir}/{args.model}_results.json"
    if args.resume:
        df_resume = pd.read_json(result_path)

    from datasets import load_dataset

    dataset = load_dataset("shangzhu/scibench", split = "train")
    """
    Dataset({
    features: ['image', 'question', 'choices', 'label', 'description', 'id'],
    num_rows: 1198
    })
    """
    

    prompts = prepare_vision_prompt(dataset)
    
    df_prompts = pd.DataFrame(prompts)

    df_results = pd.DataFrame(columns=["id", "response", "correct", 'true_label'])
    eval_bar = tqdm(df_prompts.iterrows(), total=len(df_prompts)) 

    for _, row in eval_bar:

        prompt = row["text_query"]
        image_path = row["image_path"]
        true_label = row["true_label"]


        if args.model in ['gemini', 'gpt4', 'llava', 'claude3']:
            vision_response = call_vision(prompt, image_path, args.model)
        else:
            vision_response = ""

        df_results = df_results.append({
            "question_id": row["id"],
            "response": {
                "vision": vision_response
            },
            "correct": {
                "vision": check_correctness(vision_response, true_label),
            },
            "true_label": true_label
        }, ignore_index=True)

        df_results.to_json(result_path, orient="records", indent=4)

    # df_results.to_json(result_path, orient="records", indent=4)
    # print(f"Results saved to {result_path}")

    
    
