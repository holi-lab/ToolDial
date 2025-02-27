import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_hobbies(category: str='general', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random hobby and a Wikipedia link detailing the hobby."
    category: hobby category. Possible values are:

general
sports_and_outdoors
education
collection
competition
observation
        
    """
    url = f"https://hobbies-by-api-ninjas.p.rapidapi.com/v1/hobbies"
    querystring = {}
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hobbies-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

