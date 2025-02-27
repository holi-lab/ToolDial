import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def imagine(bing_u_cookie: str, prompt: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send a GET request to imagine the parameter, and you will receive a list of generated image links. Remember, if there is any bad word in your prompt, the API will return an error message."
    bing_u_cookie: A string representing your bing _U cookie (You can get your _U cookie by simply accessing Developer Consolle and search for _U cookie Name (https://i.ibb.co/94YWpQD/1676391128.png)
        
    """
    url = f"https://bing-image-creator-api.p.rapidapi.com/imagine"
    querystring = {'bing_U_cookie': bing_u_cookie, 'prompt': prompt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bing-image-creator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

