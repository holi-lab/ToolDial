import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bet7k_aviator_latest(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will retrieve the latest results from Aviator at BET7K.
		
		- The first result is the most recent
		- The last result is the oldest"
    
    """
    url = f"https://bet7k-aviator-api.p.rapidapi.com/bet7k-aviator-latest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bet7k-aviator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

