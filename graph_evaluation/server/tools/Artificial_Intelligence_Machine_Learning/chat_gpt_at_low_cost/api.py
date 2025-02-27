import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def chat_gpt(message: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stop struggling with costly chat GPT, switch to our cost-effective API solution today. Experience smooth communication at a fraction of the price!"
    
    """
    url = f"https://chat-gpt-at-low-cost.p.rapidapi.com/chat-gpt"
    querystring = {'message': message, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chat-gpt-at-low-cost.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

