import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def australian_number_plate_to_vin(plate: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API get the VIN (Vehicle Identification Number) for a given number plate and state in Australia"
    plate: Vehicle plate number (Also know as Rego number)
        
    """
    url = f"https://australian-licence-plate-to-vin.p.rapidapi.com/"
    querystring = {'plate': plate, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "australian-licence-plate-to-vin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

