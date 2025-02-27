import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def listofsupportedbanks(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a complete list of supported banks, as well as the names of the fields needed to draw a login form similar to the bank's original."
    countrycode: Country code, ISO 3166-1 alpha-2 format (https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2). For example ES for Spain. If no code is provided, the entire list will be returned.
        
    """
    url = f"https://afterbanks-psd2-payments-and-bank-reader.p.rapidapi.com/listOfSupportedBanks/"
    querystring = {'countryCode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afterbanks-psd2-payments-and-bank-reader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categorieslist_onlyforthoseservicekeythatincludethisoption(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list with all the categories retrieved by Afterbanks own categorizer."
    countrycode: Country code, format ISO 3166-1 alpha-2 (https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2). For example ES for Spain. If no code is given, the full list is returned.
        
    """
    url = f"https://afterbanks-psd2-payments-and-bank-reader.p.rapidapi.com/transactions/categories/"
    querystring = {'countryCode': countrycode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "afterbanks-psd2-payments-and-bank-reader.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

