from openai import OpenAI
import base64
import requests
import os 
import glob 
import numpy as np 
import pdb
import pandas as pd
import json 
from tqdm import tqdm
from time import sleep
client = OpenAI()
# OpenAI API Key
api_key = os.environ['OPENAI_API_KEY']

GPT_4_TURBO_CKPT = 'gpt-4-0125-preview'
GPT_3_5_TURBO_CKPT = 'gpt-3.5-turbo-0125'
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def call_gpt4_vision(text_query, image_path, temperature=0.0, max_tokens=1024):
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": text_query
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    success = False
    while not success:
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            success = True
        except Exception as e:
            print(e)
            sleep(10)
    
    try:
        response = response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        response = ""
    return response 


def call_gpt4(text_query, system_content="You are a helpful mathematician in solving graph problems.", temperature=0.0):
    success = False
    while not success:
        try:
            response = client.chat.completions.create(
                model=GPT_4_TURBO_CKPT, #gpt-4 turbo
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": text_query + "<json>"},
                ],
                temperature = temperature
            )
            success = True
        except Exception as e:
            print(e)
            sleep(10)

    try:
        response = response.choices[0].message.content
    except Exception as e:
        print(e)
        response = ""
    
    try:
        response = json.loads(response.lower())
    except:
        response = response
    
    return response 



def call_gpt3_5(text_query, system_content="You are a helpful mathematician in solving graph problems.", temperature=0.0):
    success = False
    while not success:
        try:
            response = client.chat.completions.create(
                model=GPT_3_5_TURBO_CKPT,
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": text_query + "<json>"},
                ],
                temperature = temperature
            )
            success = True
        except Exception as e:
            print(e)
            sleep(10)

    try:
        response = response.choices[0].message.content
    except Exception as e:
        print(e)
        response = ""
    
    try:
        response = json.loads(response.lower())
    except:
        response = response
    return response 
