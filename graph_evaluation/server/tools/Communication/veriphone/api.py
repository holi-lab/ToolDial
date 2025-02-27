import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def verify(phone: str, default_country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Global phone number verification"
    phone: The phone number to verify
        default_country: The default country in a 2 letters ISO format. Example: US, RU.  Optional: the country will be infered from the prefix, from this parameter or from the IP address (in that order).
        
    """
    url = f"https://veriphone.p.rapidapi.com/verify"
    querystring = {'phone': phone, }
    if default_country:
        querystring['default_country'] = default_country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "veriphone.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def example(country_code: str='GB', type: str='mobile', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an example phone number for any country"
    country_code: The example number's country in a 2 letters ISO format. Example: US, RU.  Optional: the country will be infered from the IP address if this parameter is absent or invalid.
        type: The type of example number to return. Values: fixed_line, mobile , premium_rate, shared_cost, toll_free, voip
        
    """
    url = f"https://veriphone.p.rapidapi.com/example"
    querystring = {}
    if country_code:
        querystring['country_code'] = country_code
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "veriphone.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

