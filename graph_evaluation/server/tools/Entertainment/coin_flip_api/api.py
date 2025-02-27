import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def coin_flip(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Coin Flip API allows users to simulate a coin toss by returning a random result of "heads" or "tails". This API is useful for applications that require a random decision to be made, such as games or simulations."
    
    """
    url = f"https://coin-flip-api.p.rapidapi.com/flipcoin"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coin-flip-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

