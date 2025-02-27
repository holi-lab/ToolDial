import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def match_list(date: str, add_live: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of the match by date. Status: Played, Playing, Fixture, Cancelled"
    date: date
        add_live: is_playing
        
    """
    url = f"https://soccerway-feed.p.rapidapi.com/v1/matches/list"
    querystring = {'date': date, }
    if add_live:
        querystring['add_live'] = add_live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccerway-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_data(match_uuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match data by match uuid. Ex.: season, competition, area, form, h2h, table, lineup, commentaries, teams stats"
    match_uuid: Match uuid
        
    """
    url = f"https://soccerway-feed.p.rapidapi.com/v1/matches/data"
    querystring = {'match_uuid': match_uuid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccerway-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_list_alt(date: str, add_live: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of the match by date. Status: Played, Playing, Fixture, Cancelled"
    date: date
        add_live: is_playing
        
    """
    url = f"https://soccerway-feed.p.rapidapi.com/v1/matches/list-alt"
    querystring = {'date': date, }
    if add_live:
        querystring['add_live'] = add_live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccerway-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_multi(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search team, player by query"
    query: query
        
    """
    url = f"https://soccerway-feed.p.rapidapi.com/v1/search/multi"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccerway-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def explore_competitions(area_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of competitions by area_id"
    area_id: area_id
        
    """
    url = f"https://soccerway-feed.p.rapidapi.com/v1/explore/competitions"
    querystring = {'area_id': area_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccerway-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def explore_popular(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get popular list"
    
    """
    url = f"https://soccerway-feed.p.rapidapi.com/v1/explore/popular"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccerway-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def explore_areas(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of areas"
    
    """
    url = f"https://soccerway-feed.p.rapidapi.com/v1/explore/areas"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccerway-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competition_data(competition_id: int, season_id: int=16180, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition data"
    competition_id: competition_id
        season_id: season_id
        
    """
    url = f"https://soccerway-feed.p.rapidapi.com/v1/competitions/data"
    querystring = {'competition_id': competition_id, }
    if season_id:
        querystring['season_id'] = season_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccerway-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_squad_statistics(team_id: int, season_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Team Squad statistics"
    team_id: team_id
        season_id: season_id
        
    """
    url = f"https://soccerway-feed.p.rapidapi.com/v1/teams/squad-statistics"
    querystring = {'team_id': team_id, }
    if season_id:
        querystring['season_id'] = season_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccerway-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_photo(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player photo"
    player_id: player_id
        
    """
    url = f"https://soccerway-feed.p.rapidapi.com/v1/images/player"
    querystring = {'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccerway-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competition_image(competition_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition image"
    competition_id: competition_id
        
    """
    url = f"https://soccerway-feed.p.rapidapi.com/v1/images/competition"
    querystring = {'competition_id': competition_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccerway-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_image(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team image"
    team_id: team_id
        
    """
    url = f"https://soccerway-feed.p.rapidapi.com/v1/images/team"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccerway-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def area_flag(area_uid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get area flag by area_uid"
    area_uid: area_uid
        
    """
    url = f"https://soccerway-feed.p.rapidapi.com/v1/images/area-flag"
    querystring = {'area_uid': area_uid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccerway-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_data(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player data. Ex.: clubs career, internationals career"
    player_id: player_id
        
    """
    url = f"https://soccerway-feed.p.rapidapi.com/v1/players/data"
    querystring = {'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccerway-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_data(team_id: int, season_id: int=16180, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team data. Squad, staff, tables, matches, rankings"
    team_id: team_id
        season_id: season_id
        
    """
    url = f"https://soccerway-feed.p.rapidapi.com/v1/teams/data"
    querystring = {'team_id': team_id, }
    if season_id:
        querystring['season_id'] = season_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccerway-feed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

