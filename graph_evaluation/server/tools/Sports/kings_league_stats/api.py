import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def top_assistants(seasonid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of top assistants for an specific season"
    seasonid: The season ID. Available season IDs: *2023-s1*
        
    """
    url = f"https://kings-league-stats.p.rapidapi.com/{seasonid}/top-assistants"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kings-league-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_scorers(seasonid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of top scorers for an specific season"
    seasonid: The season ID. Available season IDs: *2023-s1*
        
    """
    url = f"https://kings-league-stats.p.rapidapi.com/{seasonid}/top-scorers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kings-league-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(seasonid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of teams foran specific season. Includes all the information about the team, its rank, its stats, etc."
    seasonid: The season ID. Available season IDs: *2023-s1*
        
    """
    url = f"https://kings-league-stats.p.rapidapi.com/{seasonid}/teams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kings-league-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players(seasonid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of players for an specific season. Includes all the public information about the player, his team, his position, his stats, etc."
    seasonid: The season ID. Available season IDs: *2023-s1*
        
    """
    url = f"https://kings-league-stats.p.rapidapi.com/{seasonid}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kings-league-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

