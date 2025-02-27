import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def head_to_head(is_id: int, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shows head to head stats and previous encounters for the home and away team of an upcoming match."
    id: (use predictions endpoint to get list of IDs)
        limit: Limit the search to only X previous encounters. (max is 10)
        
    """
    url = f"https://football-prediction-api.p.rapidapi.com/api/v2/head-to-head/{is_id}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-prediction-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def home_team_league_stats(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shows the league table and stats for the home team of an upcoming match."
    id: (use predictions endpoint to get list of IDs)
        
    """
    url = f"https://football-prediction-api.p.rapidapi.com/api/v2/home-league-stats/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-prediction-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def home_team_last_10_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Grab the statistics and list of the last 10 matches played by the home team for a certain ID"
    id: (use predictions endpoint to get list of IDs)
        
    """
    url = f"https://football-prediction-api.p.rapidapi.com/api/v2/home-last-10/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-prediction-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def away_team_last_10_matches(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Grab the statistics and list of the last 10 matches played by the away team for a certain ID"
    id: (use predictions endpoint to get list of IDs)
        
    """
    url = f"https://football-prediction-api.p.rapidapi.com/api/v2/away-last-10/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-prediction-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def away_team_league_stats(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shows the league table and stats for the away team of an upcoming match."
    id: (use predictions endpoint to get list of IDs)
        
    """
    url = f"https://football-prediction-api.p.rapidapi.com/api/v2/away-league-stats/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-prediction-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prediction_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Grab all available predictions for a match id"
    
    """
    url = f"https://football-prediction-api.p.rapidapi.com/api/v2/predictions/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-prediction-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_available_federations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of all the available federations."
    
    """
    url = f"https://football-prediction-api.p.rapidapi.com/api/v2/list-federations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-prediction-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def performance_stats_for_past_predictions(federation: str=None, market: str='classic', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns information about the accuracy of past predictions. (in the last day, 7 days, 14 days and 30 days) Can be additionally filtered by federation and market. If no market filter is provided it defaults to classic"
    federation: Filter stats by federation
        market: Show stats for a different prediction market. Defaults to "classic" if not provided
        
    """
    url = f"https://football-prediction-api.p.rapidapi.com/api/v2/performance-stats"
    querystring = {}
    if federation:
        querystring['federation'] = federation
    if market:
        querystring['market'] = market
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-prediction-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_fixture_ids(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of fixture IDs that can be used to make requests to endpoints expecting a ID url parameter.
		Can be filtered by:
		
		- iso_date
		- market
		- federation"
    
    """
    url = f"https://football-prediction-api.p.rapidapi.com/api/v2/get-list-of-fixture-ids"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-prediction-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

