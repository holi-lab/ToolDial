import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def upcoming_matches_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of Upcoming cricket matches from around the world. Any issues, queries, api integration, custom plan or to **Create your own Fantasy Sports Application**  contact us at support@techflinch.com"
    
    """
    url = f"https://fantasy-cricket.p.rapidapi.com/getcricketupcomingmatches"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def series_list_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of currently running  and upcoming cricket series from around the world. Any issues, queries, api integration, custom plan or to **Create your own Fantasy Sports Application**  contact us at support@techflinch.com"
    
    """
    url = f"https://fantasy-cricket.p.rapidapi.com/getcricketserieslist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchlist_by_seriesid(seriesid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using Series ID get the name, participating teams, start date, end date and match keys for the matches of a specified Series. Any issues, queries, api integration, custom plan or to **Create your own Fantasy Sports Application**  contact us at support@techflinch.com"
    
    """
    url = f"https://fantasy-cricket.p.rapidapi.com/getcricketmatchesbyseriesid/{seriesid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_fantasy_score(matchid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Real-time and Low-Latency Match Live Score API. Get the match score, Batting, bowling and fielding information for a specified match in real time
		Any issues, queries, api integration, custom plan or to **Create your own Fantasy Sports Application**  contact us at support@techflinch.com"
    
    """
    url = f"https://fantasy-cricket.p.rapidapi.com/getcricketfantasyscore/{matchid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fantasy_squad_api(matchid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using match ID get the list of players in the squad for the match as announced by the teams. Any issues, queries, api integration, custom plan or to **Create your own Fantasy Sports Application**  contact us at support@techflinch.com"
    
    """
    url = f"https://fantasy-cricket.p.rapidapi.com/getcricketfantasyplayers/{matchid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def playing_eleven_api(matchid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Playing Eleven by matchid when it's announced by the teams. Any issues, queries, api integration, custom plan or to **Create your own Fantasy Sports Application**  contact us at support@techflinch.com"
    
    """
    url = f"https://fantasy-cricket.p.rapidapi.com/getcricketplaying11/{matchid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_matches_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of cricket matches from around the world that are currently Live. Any issues, queries, api integration, custom plan or to **Create your own Fantasy Sports Application**  contact us at support@techflinch.com"
    
    """
    url = f"https://fantasy-cricket.p.rapidapi.com/getcricketlivematches"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fantasy-cricket.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

