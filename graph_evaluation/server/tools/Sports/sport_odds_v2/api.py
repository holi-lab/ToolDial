import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def favourite_champs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the favourite Champs Matches"
    
    """
    url = f"https://sport-odds.p.rapidapi.com/Prelive/api/Prelive/GetFavouriteChamps?skinId=wplay&lang=es-ES"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_changed_events(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the data of Changed events in Live Betting"
    
    """
    url = f"https://sport-odds.p.rapidapi.com/Live/api/Live/GetChangedEvents?skinId=wplay&lang=es-ES"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_prelive_menu(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the data of Prelive Matches."
    
    """
    url = f"https://sport-odds.p.rapidapi.com/Prelive/api/Prelive/GetSportMenu?dateFilter=&sportType=&skinId=wplay&timeZoneOffset=300&lang=es-ES"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlivemenu(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Live Menu of Live Sports."
    
    """
    url = f"https://sport-odds.p.rapidapi.com/Live/api/Live/GetLiveMenu?skinId=wplay&lang=es-ES"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sport-odds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

