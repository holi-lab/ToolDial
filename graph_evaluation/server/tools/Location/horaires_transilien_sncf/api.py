import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def offline_times(departure: str, arrival: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the offline times for a trip"
    departure: Use the 3 chars-long code for the stations
        
    """
    url = f"https://sharkoz-horaires-transilien-sncf-v1.p.rapidapi.com/gtfs/{departure}/{arrival}"
    querystring = {}
    if arrival:
        querystring['arrival'] = arrival
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sharkoz-horaires-transilien-sncf-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def last_refresh(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the timestamp of the last refresh of offline times"
    
    """
    url = f"https://sharkoz-horaires-transilien-sncf-v1.p.rapidapi.com/refresh.txt"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sharkoz-horaires-transilien-sncf-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def station_names(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the stations by name."
    
    """
    url = f"https://sharkoz-horaires-transilien-sncf-v1.p.rapidapi.com/autocomplete/{text}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sharkoz-horaires-transilien-sncf-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_times(departure: str, arrival: str='PSL', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Live train times for a designated departure and arrival stations"
    departure: Use the 3 chars-long code for the stations
        
    """
    url = f"https://sharkoz-horaires-transilien-sncf-v1.p.rapidapi.com/mobil/{departure}/{arrival}"
    querystring = {}
    if arrival:
        querystring['arrival'] = arrival
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sharkoz-horaires-transilien-sncf-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

