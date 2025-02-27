import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_item_data(item_name: str='Blink Dagger', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns JSON result with all items, that contain the requested string into the name of the item"
    
    """
    url = f"https://dota-2-items1.p.rapidapi.com/"
    querystring = {}
    if item_name:
        querystring['item_name'] = item_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dota-2-items1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

