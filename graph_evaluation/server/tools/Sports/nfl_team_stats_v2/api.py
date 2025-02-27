import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_team_receiving_data(side: str, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a JSON Response containg a list of all teams' receiving data. Parameters include whether you'd like the data for offense or defense sides, and a specific year.
		
		The side parameter should be one of these two options: "offense" or "defense"
		The year parameter should be a year between these years: 1920 - current year."
    
    """
    url = f"https://nfl-team-stats.p.rapidapi.com/v1/nfl-stats/teams/receiving-stats/{side}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nfl-team-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_team_rushing_data(year: int, side: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a JSON Response containg a list of all teams' rushingdata. Parameters include whether you'd like the data for offense or defense sides, and a specific year.
		
		The side parameter should be one of these two options: "offense" or "defense"
		The year parameter should be a year between these years: 1920 - current year."
    
    """
    url = f"https://nfl-team-stats.p.rapidapi.com/v1/nfl-stats/teams/rushing-stats/{side}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nfl-team-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_team_passing_data(year: int, side: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a JSON Response containg a list of all teams' passing data. Parameters include whether you'd like the data for offense or defense sides, and a specific year.
		
		The side parameter should be one of these two options: "offense" or "defense"
		The year parameter should be a year between these years: 1920 - current year."
    
    """
    url = f"https://nfl-team-stats.p.rapidapi.com/v1/nfl-stats/teams/passing-stats/{side}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nfl-team-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_team_win_data(year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a JSON Response containg a list of all teams' win/loss data. Parameters include a specific year.
		
		The year parameter should be a year between these years: 1920 - current year."
    
    """
    url = f"https://nfl-team-stats.p.rapidapi.com/v1/nfl-stats/teams/win-stats/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nfl-team-stats.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

