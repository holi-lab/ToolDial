import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_vehicle_details(x_anas_secret: str, username: str, vehiclenumber: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Vehicle Info By Registration number"
    
    """
    url = f"https://vehicle-rc-verification-india.p.rapidapi.com/api/getVehicleInfo/{vehiclenumber}"
    querystring = {'x-anas-secret': x_anas_secret, 'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vehicle-rc-verification-india.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

