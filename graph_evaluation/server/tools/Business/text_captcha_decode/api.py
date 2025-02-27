import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def text_captcha(link: str, isurl: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API helps in decoding Text Captcha to normal text, which can be used to bypass text captcha"
    
    """
    url = f"https://text-captcha-decode.p.rapidapi.com/imagecaptcha2"
    querystring = {'link': link, }
    if isurl:
        querystring['isurl'] = isurl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text-captcha-decode.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

