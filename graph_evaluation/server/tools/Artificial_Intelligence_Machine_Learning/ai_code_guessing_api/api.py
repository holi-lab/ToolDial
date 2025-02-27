import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def guesscode(guesscode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will allow users to guess the correct codes. The AI will respond and let you know if you are correct."
    guesscode: This endpoint will allow users to guess the correct codes. The AI will respond and let you know if you are correct.
        
    """
    url = f"https://ai-code-guessing-api.p.rapidapi.com/guessCode"
    querystring = {}
    if guesscode:
        querystring['guessCode'] = guesscode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-code-guessing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

