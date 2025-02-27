import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def binlookup(key: str, bin: str, format: str='JSON', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method helps you validate any BIN/IIN number and retrieve the full details related to the bank, brand, type, scheme, country, etc."
    key: Your API Key. Each user has a unique API Key that can be used to access the API functions. If you don't have an account yet, please create new account first.
        bin: The BIN/IIN you want to lookup/validate.
        format: Sets the format of the API response. JSON is the default format.
        
    """
    url = f"https://greip.p.rapidapi.com/BINLookup"
    querystring = {'key': key, 'bin': bin, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def badwords(key: str, text: str, scoreonly: str='no', format: str='JSON', listbadwords: str='no', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "badWords endpoint: Detects whether user inputs contain profanity or not."
    key: Your API Key. Each user has a unique API Key that can be used to access the API functions. If you don't have an account yet, please create new account first.
        text: The text you want to check.
        scoreonly: Set to `yes` to return only the score of the text and whether it's safe or not.
        format: Sets the format of the API response. JSON is the default format.
        listbadwords: Set to `yes` to list the bad-words as an Array.
        
    """
    url = f"https://greip.p.rapidapi.com/badWords"
    querystring = {'key': key, 'text': text, }
    if scoreonly:
        querystring['scoreOnly'] = scoreonly
    if format:
        querystring['format'] = format
    if listbadwords:
        querystring['listBadWords'] = listbadwords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greip.p.rapidapi.com"
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
    url = f"https://greip.p.rapidapi.com/validatePhone"
    querystring = {'countryCode': countrycode, 'phone': phone, 'key': key, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bulklookup(key: str, ips: str, params: str='currency', format: str='XML', lang: str='AR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "BulkLookup endpoint: Returns the geolocation data of multiple IP Addresses."
    key: Your API Key. Each user has a unique API Key that can be used to access the API functions. If you don't have an account yet, please create new account first.
        ips: The IP Addresses you want to lookup. It's a CSV (Comma Separated Values)
        params: The modules you want to use of the request. It's a CSV (Comma Separated Values)
        format: Sets the format of the API response. JSON is the default format.
        lang: Used to inform the API to retrieve the response in your native language.
        
    """
    url = f"https://greip.p.rapidapi.com/BulkLookup"
    querystring = {'key': key, 'ips': ips, }
    if params:
        querystring['params'] = params
    if format:
        querystring['format'] = format
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def iplookup(ip: str, key: str, format: str='XML', params: str='currency', lang: str='AR', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the geolocation data of a specific IP Address."
    ip: The IP Address you want to lookup.
        key: Your API Key. Each user has a unique API Key that can be used to access the API functions. If you don't have an account yet, please create new account first.
        format: Sets the format of the API response. JSON is the default format.
        params: The modules you want to use of the request. It's a CSV (Comma Separated Values)
        lang: Used to inform the API to retrieve the response in your native language.
        
    """
    url = f"https://greip.p.rapidapi.com/IPLookup"
    querystring = {'ip': ip, 'key': key, }
    if format:
        querystring['format'] = format
    if params:
        querystring['params'] = params
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country(countrycode: str, key: str, lang: str='AR', format: str='XML', params: str='language', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Country endpoint: Returns the information of a specific country by passing the `countrCode`."
    countrycode: The Country Code of the country you want to fetch it's data.
        key: Your API Key. Each user has a unique API Key that can be used to access the API functions. If you don't have an account yet, please create new account first.
        lang: Used to inform the API to retrieve the response in your native language.
        format: Sets the format of the API response. JSON is the default format.
        params: The modules you want to use of the request. It's a CSV (Comma Separated Values)
        
    """
    url = f"https://greip.p.rapidapi.com/Country"
    querystring = {'CountryCode': countrycode, 'key': key, }
    if lang:
        querystring['lang'] = lang
    if format:
        querystring['format'] = format
    if params:
        querystring['params'] = params
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geoip(key: str, format: str='XML', lang: str='AR', params: str='currency', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the geolocation data of the visitor."
    key: Your API Key. Each user has a unique API Key that can be used to access the API functions. If you don't have an account yet, please create new account first.
        format: Sets the format of the API response. JSON is the default format.
        lang: Used to inform the API to retrieve the response in your native language.
        params: The modules you want to use of the request. It's a CSV (Comma Separated Values)
        
    """
    url = f"https://greip.p.rapidapi.com/GeoIP"
    querystring = {'key': key, }
    if format:
        querystring['format'] = format
    if lang:
        querystring['lang'] = lang
    if params:
        querystring['params'] = params
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greip.p.rapidapi.com"
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
    url = f"https://greip.p.rapidapi.com/validateEmail"
    querystring = {'email': email, 'key': key, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def asnlookup(key: str, asn: str, islist: str='no', format: str='JSON', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ASNLookup endpoint: This method helps you lookup any AS Number. It returns the type, organisation details, routes, etc."
    key: Your API Key. Each user has a unique API Key that can be used to access the API functions. If you don't have an account yet, please create new account first.
        asn: The AS Number you want to lookup
        islist: Set this to true if you want to list all routes of both IPv4 and IPv6.
        format: Sets the format of the API response. JSON is the default format.
        
    """
    url = f"https://greip.p.rapidapi.com/ASNLookup"
    querystring = {'key': key, 'asn': asn, }
    if islist:
        querystring['isList'] = islist
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greip.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

