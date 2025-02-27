import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def now(place: str, gender: str='F', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter place and get recommendation what to wear according to current weather."
    place: The place where to get recommendation.
        gender: F or M.
        
    """
    url = f"https://what-to-wear1.p.rapidapi.com/now"
    querystring = {'place': place, }
    if gender:
        querystring['gender'] = gender
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "what-to-wear1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

