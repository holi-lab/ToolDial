import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def caiso(type: str, enddate: str='20220901', interval: str='30', startdate: str='20220101', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets standard CAISO data"
    type: Type needs to be one of \\\\\\\"co2\\\\\\\", \\\\\\\"fuelsource\\\\\\\", \\\\\\\"netdemand\\\\\\\", or \\\\\\\"all\\\\\\\". The \\\\\\\"all\\\\\\\" value will combine the results for the three other tables into a single CSV file with all the columns you requested.
        
    """
    url = f"https://caiso.p.rapidapi.com/caiso"
    querystring = {'type': type, }
    if enddate:
        querystring['enddate'] = enddate
    if interval:
        querystring['interval'] = interval
    if startdate:
        querystring['startdate'] = startdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "caiso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def caisonode(type: str, long: int=-114, radius: int=100, region: str=None, state: str='wa', name: str=None, lat: int=34, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns nodes where prices are tracked by CAISO"
    
    """
    url = f"https://caiso.p.rapidapi.com/caisonode"
    querystring = {'type': type, }
    if long:
        querystring['long'] = long
    if radius:
        querystring['radius'] = radius
    if region:
        querystring['region'] = region
    if state:
        querystring['state'] = state
    if name:
        querystring['name'] = name
    if lat:
        querystring['lat'] = lat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "caiso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def caisoregion(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets Regional Authorities where node prices are tracked by CAISO"
    
    """
    url = f"https://caiso.p.rapidapi.com/caisoregion"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "caiso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def caisoprice(type: str, page: int=12, date: str='20220601', format: str='jsonpage', node: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetches hourly CAISO price data"
    
    """
    url = f"https://caiso.p.rapidapi.com/caisoprice"
    querystring = {'type': type, }
    if page:
        querystring['page'] = page
    if date:
        querystring['date'] = date
    if format:
        querystring['format'] = format
    if node:
        querystring['node'] = node
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "caiso.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

