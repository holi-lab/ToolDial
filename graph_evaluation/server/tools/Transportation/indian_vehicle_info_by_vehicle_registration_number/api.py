import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_vehicle_info(regnumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get vehicle info by entering the required parameters."
    
    """
    url = f"https://indian-vehicle-info-by-vehicle-registration-number.p.rapidapi.com/products/Master/GetVehicleDetails"
    querystring = {'RegNumber': regnumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "indian-vehicle-info-by-vehicle-registration-number.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

