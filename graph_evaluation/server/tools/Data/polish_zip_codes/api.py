import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def find_zip_codes_by_city_and_district(accept: str, city: str, district: str, x_traceid: str='optional_abc123', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all zip codes located in city's district"
    accept: Requested content type
        x_traceid: A traceId that is copied to API response header
        
    """
    url = f"https://polish-zip-codes1.p.rapidapi.com/city/{city}/district/{district}"
    querystring = {'Accept': accept, }
    if x_traceid:
        querystring['X-TraceId'] = x_traceid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "polish-zip-codes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_zip_codes_by_city_and_street(accept: str, city: str, street: str, x_traceid: str='optional_abc123', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all zip codes by street name in the city."
    accept: Requested content type
        x_traceid: A traceId that is copied to API response header
        
    """
    url = f"https://polish-zip-codes1.p.rapidapi.com/city/{city}/street/{street}"
    querystring = {'Accept': accept, }
    if x_traceid:
        querystring['X-TraceId'] = x_traceid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "polish-zip-codes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zip_code_info(accept: str, zipcode: str, x_traceid: str='optional_abc123', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return zip code information, that is a list of zip code entries"
    accept: Requested content type
        zipcode: A requested zip code
        x_traceid: A traceId that is copied to API response header
        
    """
    url = f"https://polish-zip-codes1.p.rapidapi.com/{zipcode}"
    querystring = {'Accept': accept, }
    if x_traceid:
        querystring['X-TraceId'] = x_traceid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "polish-zip-codes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_zip_codes_by_city(accept: str, city: str, x_traceid: str='optional_abc123', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return all zip codes that belong to this city. If there is more then one city with that name, all zip codes are returned."
    accept: Requested content type
        x_traceid: A traceId that is copied to API response header
        
    """
    url = f"https://polish-zip-codes1.p.rapidapi.com/city/{city}"
    querystring = {'Accept': accept, }
    if x_traceid:
        querystring['X-TraceId'] = x_traceid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "polish-zip-codes1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

