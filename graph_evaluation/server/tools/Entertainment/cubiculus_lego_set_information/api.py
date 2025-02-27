import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def lego_set(legosetno: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get lego set by it's number, it's number that appears at each LEGO® box"
    legosetno: LEGO® set number
        
    """
    url = f"https://cubiculussets.p.rapidapi.com/lego-set/10l1g6blg76coa1o20deb7aa93q8a2ak6ad7m0k7viis14tjsnddbco24obnckb9/{legosetno}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cubiculussets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

