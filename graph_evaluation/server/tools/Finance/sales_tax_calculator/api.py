import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def supported_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of supported countries."
    
    """
    url = f"https://sales-tax-calculator2.p.rapidapi.com/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sales-tax-calculator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calculate(country: str, city: str='Meridian', zip: str='83646', street: str='936 Storey Ave', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves tax rates applicable to a specific address. This endpoint accepts address inputs to deliver up-to-date, relevant local tax rates instantly."
    country: Set to one of the countrie codes listed in Supported Countries.
        
    """
    url = f"https://sales-tax-calculator2.p.rapidapi.com/tax"
    querystring = {'country': country, }
    if city:
        querystring['city'] = city
    if zip:
        querystring['zip'] = zip
    if street:
        querystring['street'] = street
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sales-tax-calculator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

