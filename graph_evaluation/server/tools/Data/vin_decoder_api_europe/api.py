import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def vin_decoder(vin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide any VIN number for any vehicle manufactured in Europe. 
		
		Receive the Car model, maker, year, engine and other relevant information."
    vin: VIN Number to lookup.
        
    """
    url = f"https://vin-decoder-api-europe.p.rapidapi.com/vin"
    querystring = {'vin': vin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vin-decoder-api-europe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

