import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_specific_state_data_by_date(stateabbrev: str, yyyy_mm_dd: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the amount of new covid-19 cases and deaths on a specified date in a specified state.
		
		States must be specified by their Two-Letter State Abbreviations: https://www.ssa.gov/international/coc-docs/states.html"
    
    """
    url = f"https://covid-19-data-nyt.p.rapidapi.com/specific/{yyyy_mm_dd}/{stateabbrev}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-data-nyt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_specific_us_data_by_date(yyyy_mm_dd: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the amount of new covid-19 cases and deaths in the US on a specified date."
    
    """
    url = f"https://covid-19-data-nyt.p.rapidapi.com/specific/{yyyy_mm_dd}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-data-nyt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_total_state_data_by_date(yyyy_mm_dd: str, stateabbrev: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the total amount of covid-19 cases and deaths in a specified state from 2020-01-21 up to a specified date (if the data is available).
		
		States must be specified by their Two-Letter State Abbreviations: https://www.ssa.gov/international/coc-docs/states.html"
    
    """
    url = f"https://covid-19-data-nyt.p.rapidapi.com/total/{yyyy_mm_dd}/{stateabbrev}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-data-nyt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_total_us_data_by_date(yyyy_mm_dd: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the total amount of covid-19 cases and deaths in the US from 2020-01-21 up to a specified date."
    
    """
    url = f"https://covid-19-data-nyt.p.rapidapi.com/total/{yyyy_mm_dd}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-data-nyt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

