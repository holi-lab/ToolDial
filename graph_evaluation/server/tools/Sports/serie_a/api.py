import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def match_details_by_id(match_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Via this endpoint, you can retrieve the match details given a certain Match Id. Match Ids can be obtained through the Matches List endpoint. You can also query for live matches and for matches played starting from 1990."
    
    """
    url = f"https://serie-a2.p.rapidapi.com/match_stats/{match_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serie-a2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_stats_by_category(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoints allows you to retrieve players statistics accross different categories. Possible categories are: goals, shots, assists, crosses, saves, avgkmtraveled, headshotgoals , offsides,  corners , playingtime, posts-crossbars ,ballpossession]"
    category:  Possible categories are: goals, shots, assists, crosses, saves, avgkmtraveled, headshotgoals , offsides,  corners , playingtime, posts-crossbars ,ballpossession]
        
    """
    url = f"https://serie-a2.p.rapidapi.com/team_stats/{category}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serie-a2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matches_list_by_season_and_matchday(matchday: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Via this endpoint, you can retrieve all the matches that occurred on a given matchday in a given season. Note that we support season starting from **1990** till **2022**, and matchday between **1** and **38**."
    matchday: matchday>=1 and matchday<=38
        year: year>=1990 and year<=2022
        
    """
    url = f"https://serie-a2.p.rapidapi.com/match_schedule/{year}/{matchday}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serie-a2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_leaderboard(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to retrieve the current leaderboard, updated at the last match played."
    
    """
    url = f"https://serie-a2.p.rapidapi.com/leaderboard"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serie-a2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_stats_by_category(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoints allows you to retrieve players statistics accross different categories. Possible categories are: shots, goals, ballrecovery, foulsuffered, assists, keypasses, headgoals, posts, gksaves, avgkmtraveled"
    category: Possible categories are: shots, goals, ballrecovery, foulsuffered, assists, keypasses, headgoals, posts, gksaves, avgkmtraveled
        
    """
    url = f"https://serie-a2.p.rapidapi.com/player_stats/{category}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serie-a2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

