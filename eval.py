import os
import torch
import numpy as np 
import pandas as pd
import json 
from tqdm import tqdm
import argparse
import ast

from api.llava import call_llava
# from api.gpt4 import (
#     call_gpt4_vision, 
#     call_gpt4, 
#     call_gpt3_5
# )

# from api.gemini import (
#     call_gemini_pro_vision, 
#     call_gemini_pro,
#     call_palm_2
# )

# from api.claude3 import (
#     call_claude3_vision,
#     call_claude3
# )
# from api.together_api import call_together


def call_vision(text_query, image_path, model):
    if model == "gemini":
        response = call_gemini_pro_vision(text_query, image_path)
    elif model == "gpt4":
        response = call_gpt4_vision(text_query, image_path)
    elif "llava" in model:
        response = call_llava(text_query, image_path, model, extra_explanation=False)
    # elif model == "llava-13b":
    #     response = call_llava(text_query, image_path, extra_explanation=False)
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
    if not os.path.exists("images"):
        os.makedirs("images")

    for row in dataset:
        id, question, choices, label, description, img = row["id"], row["question"], row["choices"], row["label"], row["description"], row['image']
        try:

            # Convert the string to a list
            choices = ast.literal_eval(choices)
            assert len(choices)==4, "length of choices should be 4"

            prompt = f"""Given the image of a chemical compound, answer the following multiple choice question.

            Question: {question}
            Choices: 
            1. {choices[0]}
            2. {choices[1]}
            3. {choices[2]}
            4. {choices[3]}

            ONLY return a number from 1 to 4. DO NOT write the answer.
            """
            id = row["id"].split("/")[-1]
            image = f"images/{id}.png"

            #save iamges only when it's first loaded
            # img.save(image, 'png')

            prompts.append({
                "id": id,
                "text_query": prompt,
                "image_path": image,
                "true_label": label
            })
        except:
            print('Error:', id)

    return prompts


def parse_answer(response):

    response = ''.join(filter(str.isdigit, response)) 
    response = int(response)
    return response



def check_correctness(response, true_label):
    response = parse_answer(response)

    # if response == true_label:
    return response == (true_label + 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gemini", help="model name to evaluate")
    parser.add_argument('--together', action="store_true", help="use together api")
    parser.add_argument('--resume', action="store_true", help="resume from last checkpoint")
    args = parser.parse_args()
   
    result_dir = "chemistry_benchmark"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    result_path = f"{result_dir}/{args.model}_results.json"
    if args.resume:
        df_resume = pd.read_json(result_path)

    from datasets import load_dataset

    """
    Dataset({
    features: ['image', 'question', 'choices', 'label', 'description', 'id'],
    num_rows: 1198
    })
    """
    dataset = load_dataset("shangzhu/ChemQA-lite", split = "test")
    # change the ids of dataset to have only what comes after the last '/'
    dataset = dataset.map(lambda x: {"id": x["id"].split("/")[-1], "question": x["question"], "choices": x["choices"], "label": x["label"], "description": x["description"]})

    # results = pd.read_json("chemistry_benchmark/gemini_results.json")
    # dataset = dataset.filter(lambda x: x["id"] in results["question_id"].values)        


    prompts = prepare_vision_prompt(dataset)
    
    df_prompts = pd.DataFrame(prompts)

    df_results = pd.DataFrame(columns=["id", "response", "correct", 'true_label'])
    eval_bar = tqdm(df_prompts.iterrows(), total=len(df_prompts)) 

    for _, row in eval_bar:

        prompt = row["text_query"]
        image_path = row["image_path"]
        true_label = row["true_label"]

        if args.resume:

            if row["id"] in df_resume["id"].values:
                df_resume_row = df_resume[df_resume["id"] == row["id"]].iloc[0]
                print('df_resume:', df_resume_row)
                df_=pd.DataFrame({
                    "id": df_resume_row["id"],
                    "response": df_resume_row["response"],
                    "correct": df_resume_row["correct"],
                    "true_label": df_resume_row["true_label"]
                })
                df_results = pd.concat([df_results,df_]) #, ignore_index=True
                continue
                    
        # if args.model in ['gemini', 'gpt4', 'llava', 'llava-13b', 'llava-sci', 'claude3']:
        # try:
        vision_response = call_vision(prompt, image_path, args.model)
        # except:
        #     print('Error:', image_path)
        #     vision_response=""
        # else:
        #     vision_response = ""
        
        df_=pd.DataFrame({
            "id": row["id"],
            "response": vision_response,
            "correct": check_correctness(vision_response, true_label),
            "true_label": true_label
        }, index=[0])
        df_results = pd.concat([df_results,df_]) #, ignore_index=True

        df_results.to_json(result_path, orient="records", indent=4)

    
    # get final score
    print(df_results["correct"])
    vision_accuracy = [x for x in df_results["correct"]].count(True) / len(df_results)

    # save in results dir
    with open(f"{result_dir}/{args.model}_accuracy.txt", "w") as f:
        f.write(f"Vision Accuracy: {vision_accuracy}")


    
    
