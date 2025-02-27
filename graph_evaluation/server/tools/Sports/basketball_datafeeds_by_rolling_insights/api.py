import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def team_information(sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns team information"
    
    """
    url = f"https://basketball-datafeeds-by-rolling-insights.p.rapidapi.com/team-info/{sport}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_injuries(sport: str, team_id: str='28', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a record of each player's injury on a team"
    team_id: Format: One specified team ID
Team ID is available in the Team Info endpoint
        
    """
    url = f"https://basketball-datafeeds-by-rolling-insights.p.rapidapi.com/injuries/{sport}"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_schedule(date: str, sport: str, team_id: str='28', game_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns season schedule for the specified date or "now" returns current day's schedule. Daily schedule is changed at 10 AM ET."
    date: now returns current day's schedule. Daily schedule is changed at 10 AM ET.

Format: string now or YYYY-MM-DD
        team_id: Format: One specified team ID
Team ID is available in the Team Info endpoint
        
    """
    url = f"https://basketball-datafeeds-by-rolling-insights.p.rapidapi.com/schedule/{date}/{sport}"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    if game_id:
        querystring['game_id'] = game_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_season_statistics(sport: str, team_id: str='28', player_id: str='321', date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Player Season Statistics"
    team_id: Format: One specified team ID
Team ID is available in the Team Info endpoint
        date: Parameter can be omitted and request will return player stats for current season. Specify the beginning of sport season, ie: 2017-2018 season = 2017.

Format: YYYY
        
    """
    url = f"https://basketball-datafeeds-by-rolling-insights.p.rapidapi.com/player-stats/{date}/{sport}"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    if player_id:
        querystring['player_id'] = player_id
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_season_statistics(sport: str, date: str=None, team_id: str='28', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns statistics for teams"
    date: Parameter can be omitted and request will return player stats for current season. Specify the beginning of sport season, ie: 2017-2018 season = 2017.

Format: YYYY
        team_id: Format: One specified team ID
Team ID is available in the Team Info endpoint
        
    """
    url = f"https://basketball-datafeeds-by-rolling-insights.p.rapidapi.com/team-stats/{date}/{sport}"
    querystring = {}
    if date:
        querystring['date'] = date
    if team_id:
        querystring['team_id'] = team_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def weekly_schedule(date: str, sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all events from the date specified plus 7 days in advance"
    date: Returns all events from the date specified plus 7 days in advance.

Format: now or YYYY-MM-DD
        
    """
    url = f"https://basketball-datafeeds-by-rolling-insights.p.rapidapi.com/schedule-week/{date}/{sport}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_depth_charts(sport: str, team_id: str='28', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns player depth charts for a team"
    team_id: Format: One specified team ID
Team ID is available in the Team Info endpoint
        
    """
    url = f"https://basketball-datafeeds-by-rolling-insights.p.rapidapi.com/depth-charts/{sport}"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_game_data(sport: str, date: str, game_id: str=None, team_id: str='28', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides real-time game box scores."
    date: now returns current day's schedule. Daily schedule is changed at 10 AM ET.

Format: string now or YYYY-MM-DD
        team_id: Format: One specified team ID
Team ID is available in the Team Info endpoint
        
    """
    url = f"https://basketball-datafeeds-by-rolling-insights.p.rapidapi.com/live/{date}/{sport}"
    querystring = {}
    if game_id:
        querystring['game_id'] = game_id
    if team_id:
        querystring['team_id'] = team_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_information(sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns player demographic information"
    
    """
    url = f"https://basketball-datafeeds-by-rolling-insights.p.rapidapi.com/player-info/{sport}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_schedule(sport: str, date: str, team_id: str='28', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns season schedule for the specified season. If omitted, returns the schedule for current season."
    team_id: Format: One specified team ID
Team ID is available in the Team Info endpoint
        
    """
    url = f"https://basketball-datafeeds-by-rolling-insights.p.rapidapi.com/schedule-season/{date}/{sport}"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "basketball-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

