import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ip_location_of_client_visitor(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "IP Location of Client /  Visitor"
    
    """
    url = f"https://ip-geolocation-apis.p.rapidapi.com/json/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-geolocation-apis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ip_lookup_xml(ip: str, lang: str=None, callback: str=None, fields: str='status,message,continent,continentCode,country,countryCode,region,regionName,city,district,zip,lat,lon,timezone,offset,currency,isp,org,as,asname,reverse,mobile,proxy,hosting,query', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides geo information for the given IPv4/IPv6"
    lang: lang (ISO 639)	description
en	English (default)
de	Deutsch (German)
es	Español (Spanish)
pt-BR	Português - Brasil (Portuguese)
fr	Français (French)
ja	日本語 (Japanese)
zh-CN	中国 (Chinese)
ru	Русский (Russian)
        
    """
    url = f"https://ip-geolocation-apis.p.rapidapi.com/xml/{ip}"
    querystring = {}
    if lang:
        querystring['lang'] = lang
    if callback:
        querystring['callback'] = callback
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-geolocation-apis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ip_location_of_client_visitor_xml(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "IP Location of Client /  Visitor"
    
    """
    url = f"https://ip-geolocation-apis.p.rapidapi.com/xml/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-geolocation-apis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ip_lookup(ip: str, fields: str='status,message,continent,continentCode,country,countryCode,region,regionName,city,district,zip,lat,lon,timezone,offset,currency,isp,org,as,asname,reverse,mobile,proxy,hosting,query', lang: str=None, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides geo information for the given IPv4/IPv6"
    lang: lang (ISO 639)	description
en	English (default)
de	Deutsch (German)
es	Español (Spanish)
pt-BR	Português - Brasil (Portuguese)
fr	Français (French)
ja	日本語 (Japanese)
zh-CN	中国 (Chinese)
ru	Русский (Russian)
        
    """
    url = f"https://ip-geolocation-apis.p.rapidapi.com/json/{ip}"
    querystring = {}
    if fields:
        querystring['fields'] = fields
    if lang:
        querystring['lang'] = lang
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip-geolocation-apis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

