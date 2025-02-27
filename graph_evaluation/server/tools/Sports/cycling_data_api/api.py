import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_race(racecategory: str, race: str, afterstage: str='7', year: str='2022', classification: str='GC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return data on a specific race like classifications and winners"
    racecategory: Race category of the race, refer to GET Races for details of which is acceptable
        afterstage: After which stage of the race do you want the classification
        classification: Classification that you need like GC, Points, KOM, Youth, Teams and Stage
        
    """
    url = f"https://cycling-data-api.p.rapidapi.com/v1/races/{race}"
    querystring = {'raceCategory': racecategory, }
    if afterstage:
        querystring['afterStage'] = afterstage
    if year:
        querystring['year'] = year
    if classification:
        querystring['classification'] = classification
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cycling-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_races(category: str='world-tour', year: str='2022', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will give all races from the UCI calendar"
    category: Default is world-tour
Possible values are :
world-tour
world-championships
africa-tour
asia-tour
europe-tour
oceania-tour
men-junior
women-elite
women-junior
america-tour
women-world-tour
pro-series
nations-cup
        
    """
    url = f"https://cycling-data-api.p.rapidapi.com/v1/races"
    querystring = {}
    if category:
        querystring['category'] = category
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cycling-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_riders(year: str='2022', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return you all cyclists in the UCI"
    
    """
    url = f"https://cycling-data-api.p.rapidapi.com/v1/riders"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cycling-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_teams(year: str='2022', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will five you data on every teams in the UCI"
    
    """
    url = f"https://cycling-data-api.p.rapidapi.com/v1/teams"
    querystring = {}
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cycling-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

