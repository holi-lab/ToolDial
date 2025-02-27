import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def usa_plate_to_vin_decoder(license_plate: str, function: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Submit a simple HTTP get request with the license plate, and the state of the vehicle, and receive a VIN number in response. 
		
		You can the decode this VIN to vehicle details using the Free NHTSA (US Government) API here: https://vpic.nhtsa.dot.gov/api/"
    
    """
    url = f"https://usa-license-plate-to-vin-decoder.p.rapidapi.com/default/LicensePlateLambda"
    querystring = {'license_plate': license_plate, 'function': function, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "usa-license-plate-to-vin-decoder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

