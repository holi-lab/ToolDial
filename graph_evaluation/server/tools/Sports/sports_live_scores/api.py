import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tennis_rankings(n_players: str, category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns ATP or WTA Standings for a given number of players."
    n_players: Number of players to be displayed. Max: 1000.
        category: *atp* or *wta*
        
    """
    url = f"https://sports-live-scores.p.rapidapi.com/tennis/rankings/{category}/{n_players}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def handball_match_statistics(match_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Statistics about a given match_id. The match does not have to be necessarily live but also finished. You can retrieve Match IDs from the Handball Live Matches endpoint."
    match_id: You can retrieve Match IDs from the Handball Live Matches endpoint.
        
    """
    url = f"https://sports-live-scores.p.rapidapi.com/football/match_details/{match_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_match_lineups(match_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns LineUps about a given match_id. The match does not have to be necessarily live but also finished. You can retrieve Match IDs from the Basketball Live Matches endpoint."
    match_id: You can retrieve Match IDs from the BasketballLive Matches endpoint.
        
    """
    url = f"https://sports-live-scores.p.rapidapi.com/basketball/match_lineups/{match_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def futsal_league_rankings(league_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the standings for a given league ID. You can retrieve League IDs from the Futsal Live Matches endpoint."
    league_id: You can retrieve League IDs from the Futsal Live Matches endpoint.
        
    """
    url = f"https://sports-live-scores.p.rapidapi.com/futsal/rankings/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Football Live Matches with betting odds."
    
    """
    url = f"https://sports-live-scores.p.rapidapi.com/football/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_lineups(match_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns LineUps about a given match_id. The match does not have to be necessarily live but also finished. You can retrieve Match IDs from the Football Live Matches endpoint."
    match_id: You can retrieve Match IDs from the Football Live Matches endpoint.
        
    """
    url = f"https://sports-live-scores.p.rapidapi.com/football/match_lineups/{match_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_match_statistics(match_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Statistics about a given match_id. The match does not have to be necessarily live but also finished. You can retrieve Match IDs from the Football Live Matches endpoint."
    match_id: You can retrieve Match IDs from the Football Live Matches endpoint.
        
    """
    url = f"https://sports-live-scores.p.rapidapi.com/football/match_details/{match_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_league_rankings(league_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the standings for a given league ID. You can retrieve League IDs from the Cricket Live Matches endpoint."
    league_id: You can retrieve League IDs from the Cricket Live Matches endpoint.
        
    """
    url = f"https://sports-live-scores.p.rapidapi.com/cricket/rankings/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cricket_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Cricket Live Matches with betting odds"
    
    """
    url = f"https://sports-live-scores.p.rapidapi.com/cricket/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def futsal_match_statistics(match_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Statistics about a given match_id. The match does not have to be necessarily live but also finished. You can retrieve Match IDs from the Futsal Live Matches endpoint."
    match_id: You can retrieve Match IDs from the Futsal Live Matches endpoint.
        
    """
    url = f"https://sports-live-scores.p.rapidapi.com/futsal/match_details/{match_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def futsal_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Futsal Live Matches with betting odds"
    
    """
    url = f"https://sports-live-scores.p.rapidapi.com/futsal/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def esports_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Esports Live Matches with betting odds"
    
    """
    url = f"https://sports-live-scores.p.rapidapi.com/esports/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def table_tennis_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Table Tennis Live Matches with betting odds"
    
    """
    url = f"https://sports-live-scores.p.rapidapi.com/tabletennis/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def table_tennis_match_statistics(match_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Statistics about a given match_id. The match does not have to be necessarily live but also finished. You can retrieve Match IDs from the Table Tennis Live Matches endpoint."
    
    """
    url = f"https://sports-live-scores.p.rapidapi.com/tabletennis/match_details/{match_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_league_rankings(league_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the standings for a given league ID. You can retrieve League IDs from the Baseball Live Matches endpoint."
    league_id: You can retrieve League IDs from the Baseball Live Matches endpoint.
        
    """
    url = f"https://sports-live-scores.p.rapidapi.com/baseball/rankings/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_match_statistics(match_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Statistics about a given match_id. The match does not have to be necessarily live but also finished. You can retrieve Match IDs from the Baseball Live Matches endpoint."
    match_id: You can retrieve Match IDs from the Baseball Live Matches endpoint.
        
    """
    url = f"https://sports-live-scores.p.rapidapi.com/baseball/match_details/{match_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def baseball_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Baseball Live Matches with betting odds"
    
    """
    url = f"https://sports-live-scores.p.rapidapi.com/baseball/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def handball_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Handball Live Matches with betting odds"
    
    """
    url = f"https://sports-live-scores.p.rapidapi.com/handball/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_match_statistics(match_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Statistics about a given match_id. The match does not have to be necessarily live but also finished. You can retrieve Match IDs from the Football Live Matches endpoint."
    match_id: You can retrieve Match IDs from the Basketball Live Matches endpoint.
        
    """
    url = f"https://sports-live-scores.p.rapidapi.com/basketball/match_details/{match_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basketball_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Basketball Live Matches with betting odds"
    
    """
    url = f"https://sports-live-scores.p.rapidapi.com/basketball/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_match_statistics(match_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Statistics about a given match_id. The match does not have to be necessarily live but also finished. You can retrieve Match IDs from the Tennis Live Matches endpoint."
    match_id: You can retrieve Match IDs from the Tennis Live Matches endpoint.
        
    """
    url = f"https://sports-live-scores.p.rapidapi.com/tennis/match_details/{match_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tennis_live_matches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Tennis Live Matches with betting odds"
    
    """
    url = f"https://sports-live-scores.p.rapidapi.com/tennis/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def football_league_rankings(league_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the standings for a given league ID. You can retrieve League IDs from the Football Live Matches endpoint."
    league_id: You can retrieve League IDs from the Football Live Matches endpoint.
        
    """
    url = f"https://sports-live-scores.p.rapidapi.com/football/rankings/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sports-live-scores.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

