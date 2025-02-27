import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def vaidebob_aviator_latest(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will retrieve the latest results from Aviator at Vai de Bob.
		
		- The first result is the most recent
		- The last result is the oldest"
    
    """
    url = f"https://vai-de-bob-aviator-api.p.rapidapi.com/vaidebob-aviator-latest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vai-de-bob-aviator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

