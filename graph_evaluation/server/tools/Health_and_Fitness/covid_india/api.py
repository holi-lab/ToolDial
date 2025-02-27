import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def state_by_state_code(state_code: int, min_date: str=None, max_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Data For specific by its state code"
    
    """
    url = f"https://covid-india2.p.rapidapi.com/states/code/{state_code}"
    querystring = {}
    if min_date:
        querystring['min_date'] = min_date
    if max_date:
        querystring['max_date'] = max_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-india2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def india(min_date: str=None, max_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get active , recovered , deaths and vaccinated cases of India
		
		Get All Info from 1 November 2021"
    
    """
    url = f"https://covid-india2.p.rapidapi.com/country"
    querystring = {}
    if min_date:
        querystring['min_date'] = min_date
    if max_date:
        querystring['max_date'] = max_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-india2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def state_by_abbreviation(state_abbr: str, min_date: str=None, max_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Data of specific States using Abbr"
    
    """
    url = f"https://covid-india2.p.rapidapi.com/states/abbr/{state_abbr}"
    querystring = {}
    if min_date:
        querystring['min_date'] = min_date
    if max_date:
        querystring['max_date'] = max_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-india2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_states(max_date: str=None, min_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gives Data of the states"
    
    """
    url = f"https://covid-india2.p.rapidapi.com/states"
    querystring = {}
    if max_date:
        querystring['max_date'] = max_date
    if min_date:
        querystring['min_date'] = min_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-india2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

