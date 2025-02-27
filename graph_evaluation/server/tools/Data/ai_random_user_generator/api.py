import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_random_user(results: str='7', nation: str='en_US', gender: str='female', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get random user"
    
    """
    url = f"https://ai-random-user-generator.p.rapidapi.com/random-user"
    querystring = {}
    if results:
        querystring['results'] = results
    if nation:
        querystring['nation'] = nation
    if gender:
        querystring['gender'] = gender
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-random-user-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

