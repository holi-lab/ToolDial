import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_babynames(popular_only: str=None, gender: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Baby Names API endpoint. Returns 10 baby name results."
    popular_only: Whether to only return popular (top 10%) of names. Must be either true or false. If unset, default is true.
        gender: Baby name gender. Must be one of the following: boy, girl, neutral
        
    """
    url = f"https://baby-names-by-api-ninjas.p.rapidapi.com/v1/babynames"
    querystring = {}
    if popular_only:
        querystring['popular_only'] = popular_only
    if gender:
        querystring['gender'] = gender
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baby-names-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

