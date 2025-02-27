import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def plate_to_vin(plate: str='test', state: str='ca', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Resolve a vehicle VIN from a US license plate"
    
    """
    url = f"https://us-license-plate-search.p.rapidapi.com/rapidplatetovin"
    querystring = {}
    if plate:
        querystring['plate'] = plate
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-license-plate-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

