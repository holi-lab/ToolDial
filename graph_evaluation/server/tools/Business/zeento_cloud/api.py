import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def active_substances(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of Active Substances registered in EU Pesticides database"
    
    """
    url = f"https://zeento-cloud.p.rapidapi.com/pesticides/substances"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zeento-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def plufinder(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search commodity by its PLU code. Price look-up codes, commonly called PLU codes, PLU numbers, PLUs, produce codes, or produce labels, are a system of numbers that uniquely identify bulk produce sold in grocery stores and supermarkets."
    id: PLU Code, 4202 for example
        
    """
    url = f"https://zeento-cloud.p.rapidapi.com/plufinder/plu/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zeento-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

