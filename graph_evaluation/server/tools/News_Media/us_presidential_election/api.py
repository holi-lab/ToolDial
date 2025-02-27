import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def osay(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns JSON block of results for One State, All Years (OSAY), for the specified US Presidential Election year (i.e., Illinois)."
    state: US State name. No spaces.
        
    """
    url = f"https://us-presidential-election1.p.rapidapi.com/osay"
    querystring = {'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-presidential-election1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def osoy(year: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns JSON block of results for One State, One Year (OSOY), for the specified US Presidential Election year (i.e., Illinois/2008)."
    year: Election year as a string
        state: State Name or Abbreviation. Use spaces for multi-word states (i.e., New York)
        
    """
    url = f"https://us-presidential-election1.p.rapidapi.com/osoy"
    querystring = {'year': year, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-presidential-election1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def asoy(year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns JSON block of results for All States, One Year (ASOY), for the specified US Presidential Election year (i.e., 2008)."
    year: Election year. Can be either a string or a number type.
        
    """
    url = f"https://us-presidential-election1.p.rapidapi.com/asoy"
    querystring = {'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-presidential-election1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

