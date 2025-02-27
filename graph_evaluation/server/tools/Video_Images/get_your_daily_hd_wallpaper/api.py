import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def category_of_wallpaper(category: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Your choice of category for your wallpaper (for example :  cars)"
    
    """
    url = f"https://get-your-daily-hd-wallpaper.p.rapidapi.com/{category}"
    querystring = {}
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "get-your-daily-hd-wallpaper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

