import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bulklookup(ips: str, key: str, params: str='currency', format: str='XML', lang: str='AR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "BulkLookup endpoint: Returns the geolocation data of multiple IP Addresses."
    ips: The IP Addresses you want to lookup. It's a CSV (Comma Separated Values)
        key: Your API Key. Each user has a unique API Key that can be used to access the API functions. If you don't have an account yet, please create new account first.
        params: The modules you want to use of the request. It's a CSV (Comma Separated Values)
        format: Sets the format of the API response. JSON is the default format.
        lang: Used to inform the API to retrieve the response in your native language.
        
    """
    url = f"https://gre-geoip.p.rapidapi.com/BulkLookup"
    querystring = {'ips': ips, 'key': key, }
    if params:
        querystring['params'] = params
    if format:
        querystring['format'] = format
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gre-geoip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validateemail(email: str, key: str, format: str='JSON', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method can be used as an extra-layer of your system for validating email addresses."
    email: The Email Address you want to validate.
        key: Your API Key. Each user has a unique API Key that can be used to access the API functions. If you don't have an account yet, please create new account first.
        format: Sets the format of the API response. JSON is the default format.
        
    """
    url = f"https://gre-geoip.p.rapidapi.com/validateEmail"
    querystring = {'email': email, 'key': key, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gre-geoip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validatephone(countrycode: str, phone: str, key: str, format: str='JSON', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method can be used as an extra-layer of your system for validating phone numbers."
    countrycode: The ISO 3166-1 alpha-2 format of the country code of the phone number.
        phone: The Phone Number you want to validate.
        key: Your API Key. Each user has a unique API Key that can be used to access the API functions. If you don't have an account yet, please create new account first.
        format: Sets the format of the API response. JSON is the default format.
        
    """
    url = f"https://gre-geoip.p.rapidapi.com/validatePhone"
    querystring = {'countryCode': countrycode, 'phone': phone, 'key': key, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gre-geoip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def binlookup(bin: str, key: str, format: str='JSON', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method helps you validate any BIN/IIN number and retrieve the full details related to the bank, brand, type, scheme, country, etc."
    bin: The BIN/IIN you want to lookup/validate.
        key: Your API Key. Each user has a unique API Key that can be used to access the API functions. If you don't have an account yet, please create new account first.
        format: Sets the format of the API response. JSON is the default format.
        
    """
    url = f"https://gre-geoip.p.rapidapi.com/BINLookup"
    querystring = {'bin': bin, 'key': key, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gre-geoip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

