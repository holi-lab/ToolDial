import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def time_left_get(birth: str, gender: str='male', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Expected time left to live"
    birth: Birth date and year
        gender: Gender
        
    """
    url = f"https://life-left.p.rapidapi.com/time-left"
    querystring = {'birth': birth, }
    if gender:
        querystring['gender'] = gender
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "life-left.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

