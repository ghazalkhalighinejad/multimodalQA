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


from api.reka import call_reka

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
    elif "reka" in model: #reka-edge and reka-flash
        response = call_reka(text_query, model_name=model)
    elif ("llama" in args.model or args.model == "mixtral") and args.together:
        response = call_together(text_query, system_content, model=args.model)
    else:
        raise NotImplementedError
    return response


def prepare_vision_prompt(#TODO):

    image_path = #TODO
    
    text_query = #TODO

    return {
        "image_path": image_path,
        "text_query": text_query
    }


def parse_answer(response):
    # TODO
    return response



def check_correctness(response, true_label):

    # TODO
    return 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gemini", help="model name to evaluate")
    parser.add_argument('--together', action="store_true", help="use together api")
    parser.add_argument('--resume', action="store_true", help="resume from last checkpoint")
    args = parser.parse_args()

    if args.model == "llava":
        from api.llava import call_llava
   
    result_dir = "chess_benchmark"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    if args.only_fen:
        result_path = f"{result_dir}/{args.model}_results_fen.json"
    else:
        result_path = f"{result_dir}/{args.model}_results.json"
    if args.resume:
        df_resume = pd.read_json(result_path)
    
    df_prompts = pd.DataFrame(prompts)

    df_results = pd.DataFrame(columns=["chess_id", "response", "correct", 'true_label'])
    eval_bar = tqdm(df_prompts.iterrows(), total=len(df_prompts))

    gemini_results = pd.read_json("chess_benchmark/gemini_results.json")

    # load sampled_games.csv
    sampled_games = pd.read_csv("../sampled_1000.csv")       

    for _, row in eval_bar:

        if args.only_fen:
            fen_prompt = get_fen_prompt(row["id"])
            chess_id = row["id"]
            true_label = sampled_games[sampled_games['Site'] == chess_id]['Result'].values[0]

            if chess_id not in gemini_results["chess_id"].values:
                continue
        
            if args.model in ['gemini', 'gpt4', 'palm2', 'gpt3.5', 'claude3', 'reka-flash', 'reka-edge', 'llama2_7b', 'llama2_13b']:
                fen_response = call_text(fen_prompt['text_query'], system_content="You are a helpful chess player.", model=args.model)
            elif ("llama" in args.model or args.model == "mixtral") and args.together:
                fen_response = call_together(fen_prompt['text_query'], system_content="You are a helpful chess player.", model=args.model)
            else:
                fen_response = ""

            df_results = df_results.append({
                "chess_id": chess_id,
                "response": {
                    "fen": fen_response
                },
                "correct": {
                    "fen": check_correctness(fen_response, true_label)
                },
                "true_label": true_label
            }, ignore_index=True)

            df_results.to_json(result_path, orient="records", indent=4)
            eval_bar.set_description(f"Chess {chess_id} | FEN: {df_results['correct'].apply(lambda x: x['fen']).mean()*100:.2f}%")

        else:

            if args.isoscratch:
                graphical_prompt = prepare_vision_prompt(row["id"], sampled_games)
            else:
                graphical_prompt = row["graphical_prompt"]
            anl_prompt = row["anl_prompt"]
            pgn_prompt = row["pgn_prompt"]
            chess_id = row["id"]
            image_path = graphical_prompt["image_path"]

            true_label = sampled_games[sampled_games['Site'] == chess_id]['Result'].values[0]

            if args.resume:
                if chess_id in df_resume["chess_id"].values:
                    df_resume_row = df_resume[df_resume["chess_id"] == chess_id].iloc[0]
                    df_results = df_results.append({
                        "chess_id": df_resume_row['chess_id'],
                        "response": df_resume_row['response'],
                        "correct": {
                            "vision": check_correctness(df_resume_row['response']['vision'], df_resume_row['true_label']),
                            "anl": check_correctness(df_resume_row['response']['anl'], df_resume_row['true_label']),
                            "pgn": check_correctness(df_resume_row['response']['pgn'], df_resume_row['true_label']),
                            "fen": check_correctness(df_resume_row['response']['fen'], df_resume_row['true_label'])
                        },
                        "true_label": df_resume_row['true_label']
                    }, ignore_index=True)
                    continue
            if chess_id not in gemini_results["chess_id"].values:
                continue

            if args.model in ['gemini', 'gpt4', 'llava', 'claude3']:
                vision_response = call_vision(graphical_prompt['text_query'], graphical_prompt["image_path"], args.model)
            else:
                vision_response = ""
            
            if not args.isoscratch:
                if args.model in ['gemini', 'gpt4', 'palm2', 'gpt3.5', 'claude3', 'reka-flash', 'reka-edge', 'llama2_7b', 'llama2_13b']:
                    anl_response = call_text(anl_prompt['text_query'], system_content="You are a helpful chess player.", model=args.model)
                    pgn_response = call_text(pgn_prompt['text_query'], system_content="You are a helpful chess player.", model=args.model)
                elif ("llama" in args.model or args.model == "mixtral") and args.together:
                    anl_response = call_together(anl_prompt['text_query'], system_content="You are a helpful chess player.", model=args.model)
                    pgn_response = call_together(pgn_prompt['text_query'], system_content="You are a helpful chess player.", model=args.model)
                else:
                    anl_response = ""
                    pgn_response = ""
            else:
                anl_response = ""
                pgn_response = ""

            

            df_results = df_results.append({
                "chess_id": chess_id,
                "response": {
                    "vision": vision_response,
                    "anl": anl_response,
                    "pgn": pgn_response
                },
                "correct": {
                    "vision": check_correctness(vision_response, true_label),
                    "anl": check_correctness(anl_response, true_label),
                    "pgn": check_correctness(pgn_response, true_label)
                },
                "true_label": true_label
            }, ignore_index=True)

            df_results.to_json(result_path, orient="records", indent=4)
            eval_bar.set_description(f"Chess {chess_id} | Vision: {df_results['correct'].apply(lambda x: x['vision']).mean()*100:.2f}% | ANL: {df_results['correct'].apply(lambda x: x['anl']).mean()*100:.2f}% | PGN: {df_results['correct'].apply(lambda x: x['pgn']).mean()*100:.2f}%")

    df_results.to_json(result_path, orient="records", indent=4)
    print(f"Results saved to {result_path}")

    if not args.only_fen:
        scores = {
            "vision": df_results['correct'].apply(lambda x: x['vision']).mean(),
            "anl": df_results['correct'].apply(lambda x: x['anl']).mean(),
            "pgn": df_results['correct'].apply(lambda x: x['pgn']).mean()
        }
    else:
        scores = {
            "fen": df_results['correct'].apply(lambda x: x['fen']).mean()
        }
    
    if args.only_fen:
        json.dump(scores, open(f"{result_dir}/{args.model}_scores_fen.json", "w"), indent=4)
    else:
        json.dump(scores, open(f"{result_dir}/{args.model}_scores.json", "w"), indent=4)
    