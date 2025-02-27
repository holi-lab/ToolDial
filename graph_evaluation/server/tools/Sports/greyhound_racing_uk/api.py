import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def racecards(date: str='2021-06-05', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get races list.**
		
		- By default current day.
		- Or by date, setting it by parameter"
    
    """
    url = f"https://greyhound-racing-uk.p.rapidapi.com/racecards"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greyhound-racing-uk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def race_detail_info(id_race: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get race detailed info by ID {id_race}.**
		
		You can get the "id_race" from Results or Racecards endpoints"
    
    """
    url = f"https://greyhound-racing-uk.p.rapidapi.com/race/{id_race}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greyhound-racing-uk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def results(date: str='2021-06-02', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get results races by date"
    
    """
    url = f"https://greyhound-racing-uk.p.rapidapi.com/results"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "greyhound-racing-uk.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

