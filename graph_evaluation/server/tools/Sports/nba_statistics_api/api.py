import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def topscorersbyseasontotalpoints(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## Return top players by total points in descending order.
		
		_Season is specified in the URL._"
    
    """
    url = f"https://nba-statistics-api.p.rapidapi.com/api/playerdata/topscorers/total/season/2011/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-statistics-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topassistsintheplayoffsbyseasontotalassists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## Return top 20 players by assists (playoffs) in descending order.
		
		_Playoffs season is specified in the URL._"
    
    """
    url = f"https://nba-statistics-api.p.rapidapi.com/api/top_assists/playoffs/2022/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-statistics-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topassistsbyseasontotalassists(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## Return top 20 players by assists in descending order.
		
		_Season is specified in the URL._"
    
    """
    url = f"https://nba-statistics-api.p.rapidapi.com/api/top_assists/totals/2023/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-statistics-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topscorersintheplayoffsbyseasontotalpoints(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## Return top players (playoffs) by total points in descending order.
		
		_Playoffs season is specified in the URL._"
    
    """
    url = f"https://nba-statistics-api.p.rapidapi.com/api/playerdata/topscorers/playoffs/2011/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-statistics-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playerseasonstatsbyname(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "_player name is specified in the url._"
    
    """
    url = f"https://nba-statistics-api.p.rapidapi.com/api/playerdata/name/Jayson Tatum"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-statistics-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def allplayersbyseason(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "_season is specified in the url._"
    
    """
    url = f"https://nba-statistics-api.p.rapidapi.com/api/playerdata/season/2023"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-statistics-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playershotchartdataseasonplayoffs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## Return player shot chart data (season & playoffs)
		
		### Current Players Available:
		
		- LeBron James
		- James Harden
		- Stephen Curry
		    
		
		_Player name and Season are specified in the request URL._
		
		Shot chart is available [here](https://cdn.ssref.net/req/1/images/bbr/nbahalfcourt.png)."
    
    """
    url = f"https://nba-statistics-api.p.rapidapi.com/api/shot_chart_data/Lebron James/2023/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nba-statistics-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

