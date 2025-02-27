import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def test_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API is to test if server is up and running"
    
    """
    url = f"https://truecaller4.p.rapidapi.com/api/v1/test"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "truecaller4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_details(phone: int, countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This end point will display the search details for the number you are looking for."
    
    """
    url = f"https://truecaller4.p.rapidapi.com/api/v1/getDetails"
    querystring = {'phone': phone, 'countryCode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "truecaller4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_country_codes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will return list of all country codes."
    
    """
    url = f"https://truecaller4.p.rapidapi.com/api/v1/getCountryCodes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "truecaller4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_bulk_details(countrycode: str, phonenumbers: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This end point will display the search details for the bulk number's you are looking for."
    phonenumbers: Comma separated number (Without Country Code)
Example:- 9768XXXXXX,1400XXXXXX,1400XXXXXX
        
    """
    url = f"https://truecaller4.p.rapidapi.com/api/v1/getDetailsBulk"
    querystring = {'countryCode': countrycode, 'phoneNumbers': phonenumbers, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "truecaller4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

