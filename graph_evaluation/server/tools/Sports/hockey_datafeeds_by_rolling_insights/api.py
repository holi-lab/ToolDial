import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def weekly_schedule(sport: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all events from the date specified plus 7 days in advance"
    sport: Example: NHL
        date: Returns all events from the date specified plus 7 days in advance.

Format: now or YYYY-MM-DD
        
    """
    url = f"https://hockey-datafeeds-by-rolling-insights.p.rapidapi.com/schedule-week/{date}/{sport}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hockey-datafeeds-by-rolling-insights.p.rapidapi.com"
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
    sport: Example: NHL
        date: Parameter can be omitted and request will return player stats for current season. Specify the beginning of sport season, ie: 2017-2018 season = 2017.

Format: YYYY
        team_id: One single sport MUST be specified if using this parameter.

Format: One specified team ID
Team ID is available in the Team Info endpoint
        
    """
    url = f"https://hockey-datafeeds-by-rolling-insights.p.rapidapi.com/team-stats/{date}/{sport}"
    querystring = {}
    if date:
        querystring['date'] = date
    if team_id:
        querystring['team_id'] = team_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hockey-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_game_data(date: str, sport: str, team_id: str=None, game_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides real-time game box scores."
    date: now returns today's started and finished events. Specific date returns started and finished events from that date.

Format: now or YYYY-MM-DD
        sport: Example: NHL
        team_id: One single sport MUST be specified if using this parameter.

Format: One specified team ID
Team ID is available in the Team Info endpoint
        
    """
    url = f"https://hockey-datafeeds-by-rolling-insights.p.rapidapi.com/live/{date}/{sport}"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    if game_id:
        querystring['game_id'] = game_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hockey-datafeeds-by-rolling-insights.p.rapidapi.com"
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
    sport: Example: NHL
        date: Returns season schedule for the specified season. If omitted, returns the schedule for current season.
        team_id: One single sport MUST be specified if using this parameter.

Format: One specified team ID
Team ID is available in the Team Info endpoint
        
    """
    url = f"https://hockey-datafeeds-by-rolling-insights.p.rapidapi.com/schedule-season/{date}/{sport}"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hockey-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_information(sport: str, team_id: str='28', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns player demographic information"
    sport: Example: NHL
        team_id: One single sport MUST be specified if using this parameter.

Format: One specified team ID
Team ID is available in the Team Info endpoint
        
    """
    url = f"https://hockey-datafeeds-by-rolling-insights.p.rapidapi.com/player-info/{sport}"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hockey-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_information(sport: str, team_id: str='28', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns team information"
    sport: Example: NHL
        team_id: One single sport MUST be specified if using this parameter.

Format: One specified team ID
Team ID is available in the Team Info endpoint
        
    """
    url = f"https://hockey-datafeeds-by-rolling-insights.p.rapidapi.com/team-info/{sport}"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hockey-datafeeds-by-rolling-insights.p.rapidapi.com"
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
    sport: Example: NHL
        team_id: One single sport MUST be specified if using this parameter.

Format: One specified team ID
Team ID is available in the Team Info endpoint
        
    """
    url = f"https://hockey-datafeeds-by-rolling-insights.p.rapidapi.com/depth-charts/{sport}"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hockey-datafeeds-by-rolling-insights.p.rapidapi.com"
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
    sport: Example: NHL
        team_id: One single sport MUST be specified if using this parameter.

Format: One specified team ID
Team ID is available in the Team Info endpoint
        
    """
    url = f"https://hockey-datafeeds-by-rolling-insights.p.rapidapi.com/injuries/{sport}"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hockey-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_season_statistics(sport: str, team_id: str='28', date: str=None, player_id: str='488', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Player Season Statistics"
    sport: Example: NHL
        team_id: Format: One specified team ID
Team ID is available in the Team Info endpoint
        date: Parameter can be omitted and request will return player stats for current season. Specify the beginning of sport season, ie: 2017-2018 season = 2017.

Format: YYYY
        player_id: Format: One specified team ID
Player ID is available in the Player Info endpoint
        
    """
    url = f"https://hockey-datafeeds-by-rolling-insights.p.rapidapi.com/player-stats/{date}/{sport}"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    if date:
        querystring['date'] = date
    if player_id:
        querystring['player_id'] = player_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hockey-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_schedule(sport: str, date: str, team_id: str=None, game_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns season schedule for the specified date or "now" returns current day's schedule. Daily schedule is changed at 10 AM ET."
    sport: Example: NHL
        date: Returns all events from the date specified plus 7 days in advance.

Format: now or YYYY-MM-DD
        team_id: One single sport MUST be specified if using this parameter.

Format: One specified team ID
Team ID is available in the Team Info endpoint
        
    """
    url = f"https://hockey-datafeeds-by-rolling-insights.p.rapidapi.com/schedule/{date}/{sport}"
    querystring = {}
    if team_id:
        querystring['team_id'] = team_id
    if game_id:
        querystring['game_id'] = game_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hockey-datafeeds-by-rolling-insights.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

