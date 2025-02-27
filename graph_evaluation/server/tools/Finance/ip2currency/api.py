import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip2currency_api(license: str, visitorip: str=None, fromcurrencycode: str=None, tocurrencycode: str=None, fromamount: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Currency by IP Geolocation API"
    license: API license key.
        visitorip: IP address of visitor.
        fromcurrencycode: Base currency code.
        tocurrencycode: Target currency code.
        fromamount: Currency amount for conversion.
        
    """
    url = f"https://fraudlabs-ip2currency-v1.p.rapidapi.com/ip2currencywebservice.asmx"
    querystring = {'LICENSE': license, }
    if visitorip:
        querystring['VISITORIP'] = visitorip
    if fromcurrencycode:
        querystring['FROMCURRENCYCODE'] = fromcurrencycode
    if tocurrencycode:
        querystring['TOCURRENCYCODE'] = tocurrencycode
    if fromamount:
        querystring['FROMAMOUNT'] = fromamount
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fraudlabs-ip2currency-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

