import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getparcelstatus(zip_code: str, tracking_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get parcel status delivered by Yodel."
    
    """
    url = f"https://uk-delivery-tracking-all-in-one.p.rapidapi.com/Delivery/Yodel/GetParcelStatus"
    querystring = {'zip_code': zip_code, 'tracking_number': tracking_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-delivery-tracking-all-in-one.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

