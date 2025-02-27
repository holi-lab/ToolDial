import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def games(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all games included in our database"
    
    """
    url = f"https://twitch-game-popularity.p.rapidapi.com/games"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitch-game-popularity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game(name: str, year: str='2020', month: str='08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A catch all endpoint that lets you query by game, get historical data by year or by month"
    
    """
    url = f"https://twitch-game-popularity.p.rapidapi.com/game"
    querystring = {'name': name, }
    if year:
        querystring['year'] = year
    if month:
        querystring['month'] = month
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitch-game-popularity.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

