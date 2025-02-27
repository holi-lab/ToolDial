import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_salestax(state: str=None, city: str=None, zip_code: str='90210', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Sales Tax API endpoint. Returns one or more sales tax breakdowns by ZIP code according to the specified parameters. Each breakdown includes the state sales tax (if any), county sales tax (if any), city sales tax (if any), and any additional special sales taxes. All tax values are presented in decimals (e.g. 0.1 means 10% tax).
		
		Exactly one of the following must be set: zip_code or (city + state)"
    state: State name.
        city: City name.
        zip_code: Valid US ZIP code.
        
    """
    url = f"https://sales-tax-by-api-ninjas.p.rapidapi.com/v1/salestax"
    querystring = {}
    if state:
        querystring['state'] = state
    if city:
        querystring['city'] = city
    if zip_code:
        querystring['zip_code'] = zip_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sales-tax-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

