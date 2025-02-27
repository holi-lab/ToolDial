import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def holidays_province_type(province: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a list of all holidays in a specific province with a specific type
		
		For Ontario Required Parameters ON
		For Quebec Required Parameters QC
		For Nova Scotia Required Parameters NS
		For New Brunswick Required Parameters NB
		For Manitoba Required Parameters MB
		For British Columbia Required Parameters BC
		For Prince Edward Island Required Parameters PE
		For Saskatchewan Required Parameters SK
		For Alberta Required Parameters AB
		For Newfoundland and Labrador Required Parameters NL
		For Nunavut Required Parameters NU
		For Yukon Required Parameters YT
		For Northwest Territories Required Parameters NT
		
		Type: federal or provincial"
    
    """
    url = f"https://holidays-api.p.rapidapi.com/holidays/{province}/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holidays-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def holidays_province(province: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a list of all holidays, grouped by name
		
		For Ontario Required Parameters ON
		For Quebec Required Parameters QC
		For Nova Scotia Required Parameters NS
		For New Brunswick Required Parameters NB
		For Manitoba Required Parameters MB
		For British Columbia Required Parameters BC
		For Prince Edward Island Required Parameters PE
		For Saskatchewan Required Parameters SK
		For Alberta Required Parameters AB
		For Newfoundland and Labrador Required Parameters NL
		For Nunavut Required Parameters NU
		For Yukon Required Parameters YT
		For Northwest Territories Required Parameters NT"
    
    """
    url = f"https://holidays-api.p.rapidapi.com/holidays/{province}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holidays-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def holidays(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to retrieve a list of all holidays in the holidays."
    
    """
    url = f"https://holidays-api.p.rapidapi.com/holidays"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "holidays-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

