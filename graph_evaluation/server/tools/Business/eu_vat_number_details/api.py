import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_eu_vat_number_details(vat: str, lookup: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check if  the EU VAT number is valid and optionally lookup the company name and address. "
    vat: EU VAT Number
        lookup: Perform lookup on the VAT number
        
    """
    url = f"https://eu-vat-number-details.p.rapidapi.com/vies-vat-check"
    querystring = {'vat': vat, }
    if lookup:
        querystring['lookup'] = lookup
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eu-vat-number-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

