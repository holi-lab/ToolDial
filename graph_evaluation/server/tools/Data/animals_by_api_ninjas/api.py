import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_animals(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Animals API endpoint. Returns up to 10 results matching the input name parameter."
    name: common name of animal to search. This parameter supports partial matches (e.g. fox will match gray fox and red fox)
        
    """
    url = f"https://animals-by-api-ninjas.p.rapidapi.com/v1/animals"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animals-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

