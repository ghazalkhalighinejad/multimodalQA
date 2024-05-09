import reka
import re, os
from typing import Optional, Dict
from time import sleep
from functools import partial

reka.API_KEY = os.environ.get("REKA_API_KEY")

REKA_CKPT = 'reka-flash' #21b model

def call_reka(
    text_query: str, 
    image_url: Optional[str] = None, 
    temperature: Optional[float] = 0.0,
    model_name: str = REKA_CKPT,
    do_parse: bool = False
) -> str:
    if image_url is None:
        inference = partial(
            reka.chat, temperature=temperature, model_name=model_name,
        )
    else:
        inference = partial(
            reka.chat, media_url=image_url, media_type="image", temperature=temperature, model_name=model_name,
        )
    success = False
    while not success:
        try:
            response = inference(human=text_query)
            success = True
        except Exception as e:
            print(e)
            sleep(10)
    if do_parse:
        response = parse_reka_response(response)
    else:
        response = response['text'].replace('`','')
    return response


def parse_reka_response(response: Dict) -> str:
    try:
        answer = response['text'].strip().lower()
        field = answer[answer.index("answer"):]
        s = field.index(":") + 1
        field = field.replace(".", ",")
        if ',' not in field:
            e = len(field)
        else:
            e = field.index(",")
        result = field[s:e]
        result = re.sub(r"['\"]", "", result).strip()
    except Exception as e:
        result = "unknown"
    return result