import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def year_end_100_hot_songs(year: str='2021', count: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Year-End 100 Hot Songs API"
    
    """
    url = f"https://billboard-charts-rankings.p.rapidapi.com/year-end-100-hot-songs"
    querystring = {}
    if year:
        querystring['year'] = year
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-charts-rankings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hot_100(count: int=100, date: str='2021-12-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hot 100 Charts API"
    
    """
    url = f"https://billboard-charts-rankings.p.rapidapi.com/hot-100"
    querystring = {}
    if count:
        querystring['count'] = count
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-charts-rankings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_200_albums(year: str='2021', count: int=200, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Billboard 200 Albums API"
    
    """
    url = f"https://billboard-charts-rankings.p.rapidapi.com/billboard-200-albums"
    querystring = {}
    if year:
        querystring['year'] = year
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-charts-rankings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_artists(year: str='2021', count: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top Artists API"
    
    """
    url = f"https://billboard-charts-rankings.p.rapidapi.com/top-artists"
    querystring = {}
    if year:
        querystring['year'] = year
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-charts-rankings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def billboard_global_200(count: int=200, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Billboard Global 200 API"
    
    """
    url = f"https://billboard-charts-rankings.p.rapidapi.com/billboard-global-200"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "billboard-charts-rankings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

