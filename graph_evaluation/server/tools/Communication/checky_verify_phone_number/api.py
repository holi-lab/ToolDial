import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def verify(country: str, phone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To use this endpoint, you need to make an HTTP GET request to the API with the phone and country parameters in the query string."
    country: The two-letter country code of the phone number.  eg **US**, **CA**, **FR** etc.
        phone: The phone number to verify. It should be provided without any spaces or special characters.
        
    """
    url = f"https://checky-verify-phone-number.p.rapidapi.com/verify"
    querystring = {'country': country, 'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "checky-verify-phone-number.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def carrier(country: str, phone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Checky Carrier Information endpoint allows you to retrieve information about the carrier or service provider for a given phone number."
    country: The country code of the phone number.
        phone: The phone number for which to retrieve carrier information.
        
    """
    url = f"https://checky-verify-phone-number.p.rapidapi.com/carrier"
    querystring = {'country': country, 'phone': phone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "checky-verify-phone-number.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

