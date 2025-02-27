import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def extract_fast(country: str, text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find and format/validate Phone/Cell numbers in given text.
		    
		Country used to identify number format and highly recommended to set value in case you know it (as example your shop works only in US and clients have US cellnumbers).
		    
		Use 'country_auto' param in /extract endpoint case you can't provide Country and service will try to guess the country :)"
    country: Country Name or ISO2/ISO3 code
        text: Text containing phone(cell) numbers
        
    """
    url = f"https://mobile-phone-validation.p.rapidapi.com/extract_fast"
    querystring = {'country': country, 'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobile-phone-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extract(text: str, country: str=None, country_auto: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find and format/validate Phone/Cell numbers in given text.
		    
		Country used to identify number format and highly recommended to set value in case you know it (as example your shop works only in US and clients have US cellnumbers).
		    
		Use 'country_auto' param in case you can't provide Country and service will try to guess the country :). Country recognition not very fast. Read docs to explain possible values."
    text: Text containing phone(cell) numbers
        country: Country Name or ISO2/ISO3 code
        country_auto: Try identify country if not provided. 0 - false, 1- fast mode, 2 - slow mode
        
    """
    url = f"https://mobile-phone-validation.p.rapidapi.com/extract"
    querystring = {'text': text, }
    if country:
        querystring['country'] = country
    if country_auto:
        querystring['country_auto'] = country_auto
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobile-phone-validation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

