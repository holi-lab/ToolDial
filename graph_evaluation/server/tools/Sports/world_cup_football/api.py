import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_winner_by_most_wins(numvar: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get winner by   most wins.
		Get a Country with most world cup wins.
		Brazil:
		Example :
		www.football-api.uk/api/[API-KEY ]/allWiners/mostWins/1"
    
    """
    url = f"https://world-cup-football2.p.rapidapi.com/allWiners/mostWins/{numvar}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cup-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_winner_by_country_name(varteam: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get winner by team name .
		Type the name of Country, you will get all Country stats and data."
    
    """
    url = f"https://world-cup-football2.p.rapidapi.com/allWiners/team/{varteam}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cup-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_winner_by_world_cup_year(varyear: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get winner by world cup year."
    
    """
    url = f"https://world-cup-football2.p.rapidapi.com/allWiners/year/{varyear}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-cup-football2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

