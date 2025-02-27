import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def australian_vehicle_search(license_plate: str, state: str, function: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup an Australian vehicle by it's rego (license plate)"
    
    """
    url = f"https://australian-vehicle-data-nevdis.p.rapidapi.com/default/LicensePlateLambda"
    querystring = {'license_plate': license_plate, 'state': state, 'function': function, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "australian-vehicle-data-nevdis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

