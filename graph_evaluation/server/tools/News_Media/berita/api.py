import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def module(is_id: str, categories: str, publishedwithin: int, offset: int, country: str, length: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "okeee"
    
    """
    url = f"https://berita1.p.rapidapi.com/{is_id}"
    querystring = {'categories': categories, 'publishedWithin': publishedwithin, 'offset': offset, 'country': country, 'length': length, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "berita1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

