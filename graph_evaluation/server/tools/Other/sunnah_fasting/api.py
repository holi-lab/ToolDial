import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_v1_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of types"
    
    """
    url = f"https://sunnah-fasting1.p.rapidapi.com/api/v1/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sunnah-fasting1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_sources(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of sources"
    
    """
    url = f"https://sunnah-fasting1.p.rapidapi.com/api/v1/sources"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sunnah-fasting1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fastings(day: int=None, year: int=None, month: int=None, category_id: int=None, type_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of sunnah fasting"
    day: Day in month
        year: Year
        month: Month
        category_id: Category ID
        type_id: Type ID
        
    """
    url = f"https://sunnah-fasting1.p.rapidapi.com/api/v1/fastings"
    querystring = {}
    if day:
        querystring['day'] = day
    if year:
        querystring['Year'] = year
    if month:
        querystring['month'] = month
    if category_id:
        querystring['category_id'] = category_id
    if type_id:
        querystring['type_id'] = type_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sunnah-fasting1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of categories"
    
    """
    url = f"https://sunnah-fasting1.p.rapidapi.com/api/v1/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sunnah-fasting1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

