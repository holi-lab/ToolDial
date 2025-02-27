import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_team_standing(year: int=2020, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return the team standings from a certain season"
    
    """
    url = f"https://formula-18.p.rapidapi.com/teamStanding"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-18.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_driver_standing(year: int=2000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the data from the driver standing from a certain season. 
		If season has not ended yet show the current standing."
    
    """
    url = f"https://formula-18.p.rapidapi.com/driverStanding"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-18.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fastest_lap(year: int=2022, race: int=10, standing: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the fastest lap of a certain race. 
		Year is for a certain season
		Race is for a certain race
		Standing shows the driver at that certain place in the standing of fastest lap"
    
    """
    url = f"https://formula-18.p.rapidapi.com/fastestLapStanding"
    querystring = {}
    if year:
        querystring['year'] = year
    if race:
        querystring['race'] = race
    if standing:
        querystring['standing'] = standing
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-18.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_current_standing(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When a race is live shows the live standing."
    
    """
    url = f"https://formula-18.p.rapidapi.com/current"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-18.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_race_result(year: int=2022, race: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Instead of showing the standing of a race. This shows the end result of a race."
    
    """
    url = f"https://formula-18.p.rapidapi.com/raceResult"
    querystring = {}
    if year:
        querystring['year'] = year
    if race:
        querystring['race'] = race
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-18.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_race_standing(offset: int=0, race: int=2022, year: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shows the driver standing from a certain race for every lap. With their coresponding lap time"
    offset: When offset is not given offset=0. There is a limit of the amount of rows. When offset is 0 shows everything from the beginning. But when offset is 1 scroll 1 row down. So it removes the first driver and add's a driver at the bottom of the file. offset is most of the time between 0 and 200
        
    """
    url = f"https://formula-18.p.rapidapi.com/race"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if race:
        querystring['race'] = race
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-18.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

