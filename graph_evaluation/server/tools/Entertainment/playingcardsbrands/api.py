import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getlist(is_id: str=None, logo_img: str=None, brand: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full list of brands"
    
    """
    url = f"https://playingcardsbrands.p.rapidapi.com/"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if logo_img:
        querystring['logo_img'] = logo_img
    if brand:
        querystring['brand'] = brand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "playingcardsbrands.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

