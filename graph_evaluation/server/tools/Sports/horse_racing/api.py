import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def race_detail_info(id_race: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**Get racecard detailed info** and **Odds comparator**
		Horses, Jockeys, Trainers, Form, OR, Owner, Sire, Dam, Age, Weight and more information."
    
    """
    url = f"https://horse-racing.p.rapidapi.com/race/{id_race}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_races(order_by_date: str=None, date_to: str=None, class_to: int=None, distance_to: str=None, distance_from: str=None, id_horse: int=None, course: str=None, class_from: int=None, page: int=None, date_from: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The best way to search races."
    order_by_date: Results ordered by date race ascending or descending.
        class_to: Maximum race class.
        id_horse: Horse id. If you populate this field the query search races where this horse run.
        course: Like Cheltenham, Ascot, Newmarket ....
        class_from: Minimum race class.
        name: Race name or a text on the race name. Like \"Novice\", \"Handicap chase\", \"Hurdle\", ....
        
    """
    url = f"https://horse-racing.p.rapidapi.com/query-races"
    querystring = {}
    if order_by_date:
        querystring['order_by_date'] = order_by_date
    if date_to:
        querystring['date_to'] = date_to
    if class_to:
        querystring['class_to'] = class_to
    if distance_to:
        querystring['distance_to'] = distance_to
    if distance_from:
        querystring['distance_from'] = distance_from
    if id_horse:
        querystring['id_horse'] = id_horse
    if course:
        querystring['course'] = course
    if class_from:
        querystring['class_from'] = class_from
    if page:
        querystring['page'] = page
    if date_from:
        querystring['date_from'] = date_from
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def horse_stats(id_horse: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get horse results history."
    
    """
    url = f"https://horse-racing.p.rapidapi.com/horse-stats/{id_horse}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_horses(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search horses by name. 
		Once you get "*id_horse*" from this query, you can get horses stats from "**Horse stats**" endpoint."
    name: Minimum 3 characters
        
    """
    url = f"https://horse-racing.p.rapidapi.com/query-horses"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def racecards(date: str='2020-03-12', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get races list."
    
    """
    url = f"https://horse-racing.p.rapidapi.com/racecards"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def results(date: str='2020-03-13', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get results by date."
    
    """
    url = f"https://horse-racing.p.rapidapi.com/results"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jockeys_win_rate(last_days: int=7, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Jockeys stats in the last days.
		Runs, Wins and Win Rate in %."
    
    """
    url = f"https://horse-racing.p.rapidapi.com/jockeys-win-rate"
    querystring = {}
    if last_days:
        querystring['last_days'] = last_days
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trainers_win_rate(last_days: int=7, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Trainers stats in the last days.
		Runs, Wins and Win Rate in %."
    
    """
    url = f"https://horse-racing.p.rapidapi.com/trainers-win-rate"
    querystring = {}
    if last_days:
        querystring['last_days'] = last_days
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "horse-racing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

