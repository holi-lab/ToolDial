import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def https_www_agora_io_product(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Agora.io simple SDK allows you to integrate call functionality into your app or website effortlessly. From documentation to integration, the Agora.io API is straightforward and built on simplicity. Adding a few lines of code gives limitless communication possibilities to your users."
    
    """
    url = f"https://chris_agora-agora-io-v1.p.rapidapi.com/https://www.agora.io/product/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chris_agora-agora-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

