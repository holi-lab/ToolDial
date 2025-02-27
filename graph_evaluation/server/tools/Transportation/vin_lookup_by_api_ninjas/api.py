import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_vinlookup(vin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas VIN Lookup API endpoint. Returns key vehicle information including manufacturer, country of origin and model year for a given VIN."
    vin: valid VIN to check. Must be a 17-character string.
        
    """
    url = f"https://vin-lookup-by-api-ninjas.p.rapidapi.com/v1/vinlookup"
    querystring = {'vin': vin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vin-lookup-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

