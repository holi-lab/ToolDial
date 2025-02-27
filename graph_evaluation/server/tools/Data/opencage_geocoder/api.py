import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geocode_v1_json(q: str, key: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "geocode an address"
    q: "latitude,longitude" or "address" (without ")
        key: The API key you got when registering on https://geocoder.opencagedata.com/
        language: an IETF format language code (such as es for Spanish or pt-BR for Brazilian Portuguese); if this is omitted a code of en (English) will be assumed
        
    """
    url = f"https://opencage-geocoder.p.rapidapi.com/geocode/v1/json"
    querystring = {'q': q, 'key': key, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opencage-geocoder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

