import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latest_rates(base_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the latest exchange rates for the base currency code you have supplied.
		
		You can view the list of supported currency codes here: [https://www.exchangerate-api.com/docs/supported-currencies](https://www.exchangerate-api.com/docs/supported-currencies)"
    
    """
    url = f"https://exchangerate-api.p.rapidapi.com/rapid/latest/{base_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "exchangerate-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

