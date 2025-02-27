import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def vin_decoder(vin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Accessing the VIN decoder will allow you access to 36 different datapoints related to vehicle information, vehicle features, and manufacturer information. You must provide a 17 character VIN. The Warranty Experts VIN decoder is capable of correcting invalid VINs, but it is advised to be as accurate as possible."
    
    """
    url = f"https://vin-decoder28.p.rapidapi.com/api/v1/rapid-api-vin-decoder"
    querystring = {'vin': vin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vin-decoder28.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

