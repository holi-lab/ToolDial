import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stickers_trending(include: str='STICKER_PACK', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API that provides information on creators and stickerpacks of 16 popular stickers"
    include: Use when you want to receive a response including sticker pack or creator information
        
    """
    url = f"https://mojitok-sticker-store-sdk.p.rapidapi.com/stickers/trending"
    querystring = {}
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mojitok-sticker-store-sdk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

