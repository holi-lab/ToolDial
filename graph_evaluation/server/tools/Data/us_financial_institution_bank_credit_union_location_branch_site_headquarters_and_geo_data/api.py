import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_institution_by_zip_code_postal(filter_value: str, page: int, limit: int, filter_column: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use the state i.e. "90024" or "68135" to get Bank / Credit Union branches with a specific zip code or postal."
    
    """
    url = f"https://us-financial-institution-bank.p.rapidapi.com/"
    querystring = {'filter_value': filter_value, 'page': page, 'limit': limit, 'filter_column': filter_column, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-financial-institution-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_institution_by_city(page: int, filter_value: str, filter_column: str, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use the state i.e. "Dallas" or "Los+Angeles" full name to get Bank / Credit Union branches with a specific city. Note the results also include and will say if the given branch is the headquarters or not as well."
    
    """
    url = f"https://us-financial-institution-bank.p.rapidapi.com/"
    querystring = {'page': page, 'filter_value': filter_value, 'filter_column': filter_column, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-financial-institution-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_institution_by_state(page: int, filter_value: str, filter_column: str, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use the state i.e. "TX" or "NE" or "CO" abbreviation to get Bank / Credit Union branches in a state. Note the results also include and will say if the given branch is the headquarters or not as well."
    
    """
    url = f"https://us-financial-institution-bank.p.rapidapi.com/"
    querystring = {'page': page, 'filter_value': filter_value, 'filter_column': filter_column, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-financial-institution-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_instituiton_by_routing(filter_value: str, filter_column: str, page: int, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a Credit Union or Bank by Routing # i.e.   104900116"
    
    """
    url = f"https://us-financial-institution-bank.p.rapidapi.com/"
    querystring = {'filter_value': filter_value, 'filter_column': filter_column, 'page': page, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-financial-institution-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_institution_by_rssd_id(filter_value: int, filter_column: str, limit: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a Bank or Credit Union by it's RSSD ID i.e.  1155 and returns it's branch locations as well if they share the same  RSSD ID."
    
    """
    url = f"https://us-financial-institution-bank.p.rapidapi.com/?page=1&limit=10&filter_column=idrssd&filter_value=1155"
    querystring = {'filter_value': filter_value, 'filter_column': filter_column, 'limit': limit, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-financial-institution-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_instituiton_by_domain(filter_value: str, filter_column: str, limit: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a Credit Union or Bank by it's website or domain (please enter without www) i.e. usbank.com will be acceptable."
    
    """
    url = f"https://us-financial-institution-bank.p.rapidapi.com/"
    querystring = {'filter_value': filter_value, 'filter_column': filter_column, 'limit': limit, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-financial-institution-bank.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

