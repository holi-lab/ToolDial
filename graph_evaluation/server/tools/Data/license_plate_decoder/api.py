import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def plate_lookup(licenseplate: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search through our database for License Plate Data, provide a license plate and state and we will respond with VIN and other vehicle specifications!"
    
    """
    url = f"https://license-plate-decoder.p.rapidapi.com/plate"
    querystring = {'licensePlate': licenseplate, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "license-plate-decoder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

