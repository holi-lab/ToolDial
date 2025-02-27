import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a list of countries compatible with the phone number validator"
    
    """
    url = f"https://ag4memnon-international-phone-number-validation-v1.p.rapidapi.com/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ag4memnon-international-phone-number-validation-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def phone_number(country_code_alpha2: str, phone_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Takes in a phone number and 2 character alpha country code (which can be retrieved from the countries endpoint) and returns a self-describing object describing the number. The API returns if the number is possible, as well as some geo-location data on where the number may be located.  The API also returns the number in all popular ISO formats."
    
    """
    url = f"https://ag4memnon-international-phone-number-validation-v1.p.rapidapi.com/validation/phone_number?country_code_alpha2={country_code_alpha2}&phone_number={phone_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ag4memnon-international-phone-number-validation-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

