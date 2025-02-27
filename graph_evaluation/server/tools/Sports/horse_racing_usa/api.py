import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def race(id_race: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get race detailed info by ID {id_race}. 
		Runners and riders for each race."
    
    """
    url = f"https://horse-racing-usa.p.rapidapi.com/race/{id_race}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def racecards(date: str='2021-03-18', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get races list."
    
    """
    url = f"https://horse-racing-usa.p.rapidapi.com/racecards"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def results(date: str='2021-03-18', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get results by date."
    
    """
    url = f"https://horse-racing-usa.p.rapidapi.com/results"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def horse_stats(id_horse: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get horses result history. You can find {id_horse} on "/race/XXX" endpoint."
    
    """
    url = f"https://horse-racing-usa.p.rapidapi.com/horse-stats/{id_horse}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainers_win_rate(last_days: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Trainers stats in the last days.
		Runs, Wins and Win Rate in %."
    
    """
    url = f"https://horse-racing-usa.p.rapidapi.com/trainers-win-rate"
    querystring = {}
    if last_days:
        querystring['last_days'] = last_days
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jockeys_win_rate(last_days: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Jockeys stats in the last days.
		Runs, Wins and Win Rate in %."
    
    """
    url = f"https://horse-racing-usa.p.rapidapi.com/jockeys-win-rate"
    querystring = {}
    if last_days:
        querystring['last_days'] = last_days
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def horses_win_rate(last_days: str='30', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Horses stats in the last days.
		Runs, Wins and Win Rate in %."
    
    """
    url = f"https://horse-racing-usa.p.rapidapi.com/horses-win-rate"
    querystring = {}
    if last_days:
        querystring['last_days'] = last_days
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing-usa.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

