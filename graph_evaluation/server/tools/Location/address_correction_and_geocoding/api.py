import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def address(addressline1: str, addressline2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Address"
    addressline1: First line of address
        addressline2: Second line of address - city, state, zip.
        
    """
    url = f"https://yurisw-address-correction-and-geocoding.p.rapidapi.com/address"
    querystring = {'AddressLine1': addressline1, 'AddressLine2': addressline2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yurisw-address-correction-and-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

