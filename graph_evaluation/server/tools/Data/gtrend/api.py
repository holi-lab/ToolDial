import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def regionlist(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List support region / Country"
    
    """
    url = f"https://gtrend.p.rapidapi.com/region"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gtrend.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geomap(property: str, resolution: str, geo: str, timezone: str, dataframe: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get geo map data from given query"
    geo: for supported region/country , please see regionList from region menu
        timezone: support timezone from GMT-12 to GMT+14

        
    """
    url = f"https://gtrend.p.rapidapi.com/geo-map"
    querystring = {'property': property, 'resolution': resolution, 'geo': geo, 'timezone': timezone, 'dataframe': dataframe, 'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gtrend.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def relatedquery(timezone: str, query: str, dataframe: str, property: str, geo: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get related query from given query"
    timezone: support timezone from GMT-12 to GMT+14
        dataframe: h = hour
d = day
m = month
y = year
        geo: for supported region/country , please see regionList from region menu
        
    """
    url = f"https://gtrend.p.rapidapi.com/related-query"
    querystring = {'timezone': timezone, 'query': query, 'dataframe': dataframe, 'property': property, 'geo': geo, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gtrend.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categorylist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://gtrend.p.rapidapi.com/category-list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gtrend.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timezonelist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://gtrend.p.rapidapi.com/timezone"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gtrend.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def relatedtopic(query: str, dataframe: str, timezone: str, geo: str, property: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get related topic from given query"
    timezone: support timezone from GMT-12 to GMT+14
        geo: for supported region/country , please see regionList from region menu
        
    """
    url = f"https://gtrend.p.rapidapi.com/related-topic"
    querystring = {'query': query, 'dataframe': dataframe, 'timezone': timezone, 'geo': geo, 'property': property, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gtrend.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timeseries(query: str, geo: str, dataframe: str, property: str, timezone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get timeseries from query"
    geo: for supported region/country , please see regionList from region menu
        dataframe: h = hour
d = day
m = month
y = year
        timezone: support timezone from GMT-12 to GMT+14
        
    """
    url = f"https://gtrend.p.rapidapi.com/time-series"
    querystring = {'query': query, 'geo': geo, 'dataframe': dataframe, 'property': property, 'timezone': timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gtrend.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

