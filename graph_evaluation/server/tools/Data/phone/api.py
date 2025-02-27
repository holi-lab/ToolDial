import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def area_code(country: str, region: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the International Country Calling Code and the Local Area Code information"
    country: "ISO 2 letter country code" or "Name of the Country in English"
        region: Geographic Region (e.g. City). No accented letter is required.
        
    """
    url = f"https://metropolis-api-phone.p.rapidapi.com/area-code"
    querystring = {'country': country, 'region': region, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metropolis-api-phone.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def analysis(telephone: str, country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Analyse, validate and parse the content of a given Phone Number"
    telephone: Full Telephone Number
        country: "ISO 2 letter country code" or "Name of the Country in English"
        
    """
    url = f"https://metropolis-api-phone.p.rapidapi.com/analysis"
    querystring = {'telephone': telephone, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metropolis-api-phone.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flag(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Flag Image from a given Country"
    country: "Country ISO Code" or "Description of the Country in English"
        
    """
    url = f"https://metropolis-api-phone.p.rapidapi.com/flag"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metropolis-api-phone.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def iso(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the ISO codes from a given Country"
    country: Description of the Country in English
        
    """
    url = f"https://metropolis-api-phone.p.rapidapi.com/iso"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metropolis-api-phone.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def region(country: str, area_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Geographic Region information"
    country: "ISO 2 letter country code" or "Name of the Country in English"
        area_code: Local Area Code
        
    """
    url = f"https://metropolis-api-phone.p.rapidapi.com/region"
    querystring = {'country': country, 'area-code': area_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metropolis-api-phone.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def directory(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the Local Area Codes from a given Country"
    country: "ISO 2 letter country code" or "Name of the Country in English"
        
    """
    url = f"https://metropolis-api-phone.p.rapidapi.com/directory"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "metropolis-api-phone.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

