import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_uuid(quantity: str='25', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates a UUID and returns to client"
    quantity: Used to determine how many UUIDs you want in the response. Maximum of 50 uuids per request.
        
    """
    url = f"https://uuid-generator5.p.rapidapi.com/random-uuid"
    querystring = {}
    if quantity:
        querystring['quantity'] = quantity
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uuid-generator5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

