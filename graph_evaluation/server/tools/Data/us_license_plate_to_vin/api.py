import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def us_license_plate_to_vin(plate: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup the full VIN from vehicle license plates. Support license plates across all 50 states in the USA. Supports cars, trucks, motorcycles, RVs, and more."
    state: Needs to be two letter state abbreviations. For example: CA, NV
        
    """
    url = f"https://us-license-plate-to-vin.p.rapidapi.com/licenseplate"
    querystring = {'plate': plate, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-license-plate-to-vin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

