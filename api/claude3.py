import anthropic
import os 
import base64
from PIL import Image
import pdb
from time import sleep
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

CLAUDE_CKPT = "claude-3-opus-20240229"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_claude3_vision(text_query, image_path, temperature=0.0):
    base64_image = encode_image(image_path)
    success = False
    while not success:
        try:
            response = client.messages.create(
                model=CLAUDE_CKPT,
                max_tokens=1024,
                temperature=temperature,
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                                }
                            },
                            {
                                "type": "text", 
                                "text": text_query
                            }
                        ]
                    }
                ]
            )
            response = response.content[0].text
            success = True
        except Exception as e:
            print(e)
            sleep(60)
    return response

    

def call_claude3(text_query, system_content, temperature=0.0):
    success = False
    while not success:
        try:
            response = client.messages.create(
                model=CLAUDE_CKPT,
                max_tokens=1024,
                temperature=temperature,
                system=system_content,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text_query
                            }
                        ]
                    }
                ]
            )
            response = response.content[0].text
            success = True
        except Exception as e:
            print(e)
            sleep(60)
    return response
