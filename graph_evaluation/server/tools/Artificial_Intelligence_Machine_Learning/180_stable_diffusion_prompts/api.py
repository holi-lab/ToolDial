import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_the_prompts(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We understand the importance of stable diffusion and have created Stable Diffusion Prompts to help you achieve it. In Stable Diffusion Prompts can improve your experience and help you achieve better stability."
    
    """
    url = f"https://180-stable-diffusion-prompts.p.rapidapi.com/prompts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "180-stable-diffusion-prompts.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

