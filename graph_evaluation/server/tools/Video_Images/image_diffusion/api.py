import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def image_stable_diffusion(prompt: str, ratio: str, style: str='CREATIVE', json: bool=None, hiresfix: bool=None, negative_prompt: str=None, cfg: str='7.5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Most stable and fast diffusion to generate safe image `stuff.`"
    prompt: Describe your fantasy in prompt
        ratio: Available ratio.

> 1:1
> 9:16
> 16:9
> 4:3
> 3:2
> 2:3
> 5:4
> 4:5
> 8:10
> 3:1
> 3:4
        style: Available styles
> **CREATIVE**
>> default style is **CREATIVE** 

> **REALISTIC**
> **ANIME**
        json: Get a json response instead of image buffer 
        hiresfix: Enable High Resolution Fix

        negative_prompt: Describe negative prompt as you want.
        cfg: CFG Scale

> Min 3.0
>  Max 16.0
        
    """
    url = f"https://image-diffusion.p.rapidapi.com/image/stable/diffusion"
    querystring = {'prompt': prompt, 'ratio': ratio, }
    if style:
        querystring['style'] = style
    if json:
        querystring['json'] = json
    if hiresfix:
        querystring['hiresFix'] = hiresfix
    if negative_prompt:
        querystring['negative_prompt'] = negative_prompt
    if cfg:
        querystring['cfg'] = cfg
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-diffusion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_anime_diffusion(steps: int, image_num: int, width: int, cfg: int, height: int, prompt: str, negative_prompt: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "design to generate weeb stuff."
    steps: maximum 50
        image_num: maximum 4
        width: maximum width/height 1024
        cfg: maximum 20
        height: maximum width/height 1024
        prompt: Describe your fantasy in prompt
        negative_prompt: negative prompt. v2 dont need this parameter.
        
    """
    url = f"https://image-diffusion.p.rapidapi.com/image/anime/diffusion"
    querystring = {'steps': steps, 'image_num': image_num, 'width': width, 'cfg': cfg, 'height': height, 'prompt': prompt, }
    if negative_prompt:
        querystring['negative_prompt'] = negative_prompt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-diffusion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_stable_prompter(url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "generate prompt from image url for stable diffusion"
    url: full path image url.
        
    """
    url = f"https://image-diffusion.p.rapidapi.com/image/stable/prompter"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "image-diffusion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

