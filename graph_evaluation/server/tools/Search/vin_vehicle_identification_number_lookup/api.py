import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_vin_vin_vin(vin: str='WP0CA29873U624034', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter the Vehicle Identification Number for Search"
    
    """
    url = f"https://vin-vehicle-identification-number-lookup.p.rapidapi.com/search_vin"
    querystring = {}
    if vin:
        querystring['vin'] = vin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vin-vehicle-identification-number-lookup.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

