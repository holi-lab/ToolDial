import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def captcha_generator(fontname: str='sora', noise_number: int=10, text: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a captcha generator tool that gives you an image of 270x80 pixels and the text solution. It has different settings that allows you to configure the captcha difficulty, and even create a captcha with a custom text solution."
    fontname: Supported fontnames:  'sora', 'noto-sans', 'ubuntu'. Default = 'sora'
        noise_number: An integer value. Default = 10
        text: A custom captcha solution text. Default text solution is set aleatory
        
    """
    url = f"https://captcha-generator.p.rapidapi.com/"
    querystring = {}
    if fontname:
        querystring['fontname'] = fontname
    if noise_number:
        querystring['noise_number'] = noise_number
    if text:
        querystring['text'] = text
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "captcha-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

