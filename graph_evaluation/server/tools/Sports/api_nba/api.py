import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def players_statistics_per_game_id(team: int=None, game: int=8133, is_id: int=None, season: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics from one or more players.
		
		> This endpoint requires at least one parameter."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/players/statistics"
    querystring = {}
    if team:
        querystring['team'] = team
    if game:
        querystring['game'] = game
    if is_id:
        querystring['id'] = is_id
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings_per_team_and_season(league: str='standard', conference: str=None, team: int=1, division: str=None, season: int=2021, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the standings for a league & season.
		
		**Available conferences**
		* East
		* West
		
		**Available divisions**
		* Atlantic
		* Central
		* Northwest
		* Pacific
		* Southeast
		* Southwest
		
		> Parameters `league` & `season` are required."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/standings"
    querystring = {}
    if league:
        querystring['league'] = league
    if conference:
        querystring['conference'] = conference
    if team:
        querystring['team'] = team
    if division:
        querystring['division'] = division
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings_per_division_and_season(team: int=None, conference: str=None, season: int=2021, division: str='southeast', league: str='standard', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the standings for a league & season.
		
		**Available conferences**
		* East
		* West
		
		**Available divisions**
		* Atlantic
		* Central
		* Northwest
		* Pacific
		* Southeast
		* Southwest
		
		> Parameters `league` & `season` are required."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/standings"
    querystring = {}
    if team:
        querystring['team'] = team
    if conference:
        querystring['conference'] = conference
    if season:
        querystring['season'] = season
    if division:
        querystring['division'] = division
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players(search: str=None, is_id: int=None, season: int=2021, country: str=None, name: str=None, team: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about players.
		
		The player `id` are **unique** in the API and players keep it among all seasons.
		
		> This endpoint requires at least one parameter."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/players"
    querystring = {}
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    if season:
        querystring['season'] = season
    if country:
        querystring['country'] = country
    if name:
        querystring['name'] = name
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_per_id(division: str=None, name: str=None, conference: str=None, code: str=None, is_id: int=1, search: str=None, league: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about teams.
		
		The team `id` are **unique** in the API and teams keep it among all seasons.
		
		**Available conferences**
		* East
		* West
		
		**Available divisions**
		* Atlantic
		* Central
		* Northwest
		* Pacific
		* Southeast
		* Southwest"
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/teams"
    querystring = {}
    if division:
        querystring['division'] = division
    if name:
        querystring['name'] = name
    if conference:
        querystring['conference'] = conference
    if code:
        querystring['code'] = code
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_per_team_and_season(date: str=None, h2h: str=None, team: int=1, is_id: int=None, live: str=None, season: int=2021, league: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of games according to the parameters.
		
		**Available status**
		* 1 : Not Started
		* 2 : live
		* 3 : Finished
		
		> This endpoint requires at least one parameter."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/games"
    querystring = {}
    if date:
        querystring['date'] = date
    if h2h:
        querystring['h2h'] = h2h
    if team:
        querystring['team'] = team
    if is_id:
        querystring['id'] = is_id
    if live:
        querystring['live'] = live
    if season:
        querystring['season'] = season
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_per_season(is_id: int=None, season: int=2021, date: str=None, live: str=None, team: int=None, h2h: str=None, league: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of games according to the parameters.
		
		**Available status**
		* 1 : Not Started
		* 2 : live
		* 3 : Finished
		
		> This endpoint requires at least one parameter."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/games"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if season:
        querystring['season'] = season
    if date:
        querystring['date'] = date
    if live:
        querystring['live'] = live
    if team:
        querystring['team'] = team
    if h2h:
        querystring['h2h'] = h2h
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_per_id(team: int=None, league: str=None, is_id: int=8899, season: int=None, h2h: str=None, date: str=None, live: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of games according to the parameters.
		
		**Available status**
		* 1 : Not Started
		* 2 : live
		* 3 : Finished
		
		> This endpoint requires at least one parameter."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/games"
    querystring = {}
    if team:
        querystring['team'] = team
    if league:
        querystring['league'] = league
    if is_id:
        querystring['id'] = is_id
    if season:
        querystring['season'] = season
    if h2h:
        querystring['h2h'] = h2h
    if date:
        querystring['date'] = date
    if live:
        querystring['live'] = live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_between_two_teams(team: int=None, live: str=None, season: int=None, is_id: int=None, date: str=None, h2h: str='1-2', league: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of games according to the parameters.
		
		**Available status**
		* 1 : Not Started
		* 2 : live
		* 3 : Finished
		
		> This endpoint requires at least one parameter."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/games"
    querystring = {}
    if team:
        querystring['team'] = team
    if live:
        querystring['live'] = live
    if season:
        querystring['season'] = season
    if is_id:
        querystring['id'] = is_id
    if date:
        querystring['date'] = date
    if h2h:
        querystring['h2h'] = h2h
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_statistics_per_player_and_season(game: int=None, season: int=2020, is_id: int=236, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics from one or more players.
		
		> This endpoint requires at least one parameter."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/players/statistics"
    querystring = {}
    if game:
        querystring['game'] = game
    if season:
        querystring['season'] = season
    if is_id:
        querystring['id'] = is_id
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_statistics_per_team_and_season(season: int=2020, game: int=None, team: int=1, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics from one or more players.
		
		> This endpoint requires at least one parameter."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/players/statistics"
    querystring = {}
    if season:
        querystring['season'] = season
    if game:
        querystring['game'] = game
    if team:
        querystring['team'] = team
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_players(season: int=None, search: str='james', country: str=None, is_id: int=None, team: int=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search a player by lastname."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/players"
    querystring = {}
    if season:
        querystring['season'] = season
    if search:
        querystring['search'] = search
    if country:
        querystring['country'] = country
    if is_id:
        querystring['id'] = is_id
    if team:
        querystring['team'] = team
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_teams(code: str=None, league: str=None, division: str=None, search: str='atl', name: str=None, is_id: int=None, conference: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search team by name."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/teams"
    querystring = {}
    if code:
        querystring['code'] = code
    if league:
        querystring['league'] = league
    if division:
        querystring['division'] = division
    if search:
        querystring['search'] = search
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    if conference:
        querystring['conference'] = conference
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings_per_conference_and_season(team: int=None, league: str='standard', division: str=None, season: int=2021, conference: str='east', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the standings for a league & season.
		
		**Available conferences**
		* East
		* West
		
		**Available divisions**
		* Atlantic
		* Central
		* Northwest
		* Pacific
		* Southeast
		* Southwest
		
		> Parameters `league` & `season` are required."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/standings"
    querystring = {}
    if team:
        querystring['team'] = team
    if league:
        querystring['league'] = league
    if division:
        querystring['division'] = division
    if season:
        querystring['season'] = season
    if conference:
        querystring['conference'] = conference
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_per_country(season: int=None, name: str=None, is_id: int=None, team: int=None, country: str='spain', search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about players.
		
		The player `id` are **unique** in the API and players keep it among all seasons.
		
		> This endpoint requires at least one parameter."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/players"
    querystring = {}
    if season:
        querystring['season'] = season
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    if team:
        querystring['team'] = team
    if country:
        querystring['country'] = country
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_per_id(search: str=None, season: int=None, name: str=None, country: str=None, team: int=None, is_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about players.
		
		The player `id` are **unique** in the API and players keep it among all seasons.
		
		> This endpoint requires at least one parameter."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/players"
    querystring = {}
    if search:
        querystring['search'] = search
    if season:
        querystring['season'] = season
    if name:
        querystring['name'] = name
    if country:
        querystring['country'] = country
    if team:
        querystring['team'] = team
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_per_team_and_season(season: int=2021, country: str=None, name: str=None, search: str=None, team: int=1, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about players.
		
		The player `id` are **unique** in the API and players keep it among all seasons.
		
		> This endpoint requires at least one parameter."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/players"
    querystring = {}
    if season:
        querystring['season'] = season
    if country:
        querystring['country'] = country
    if name:
        querystring['name'] = name
    if search:
        querystring['search'] = search
    if team:
        querystring['team'] = team
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_per_code(league: str=None, is_id: int=None, code: str='ATL', name: str=None, division: str=None, search: str=None, conference: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about teams.
		
		The team `id` are **unique** in the API and teams keep it among all seasons.
		
		**Available conferences**
		* East
		* West
		
		**Available divisions**
		* Atlantic
		* Central
		* Northwest
		* Pacific
		* Southeast
		* Southwest"
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/teams"
    querystring = {}
    if league:
        querystring['league'] = league
    if is_id:
        querystring['id'] = is_id
    if code:
        querystring['code'] = code
    if name:
        querystring['name'] = name
    if division:
        querystring['division'] = division
    if search:
        querystring['search'] = search
    if conference:
        querystring['conference'] = conference
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_per_division(name: str=None, league: str=None, is_id: int=None, code: str=None, division: str='Southeast', search: str=None, conference: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about teams.
		
		The team `id` are **unique** in the API and teams keep it among all seasons.
		
		**Available conferences**
		* East
		* West
		
		**Available divisions**
		* Atlantic
		* Central
		* Northwest
		* Pacific
		* Southeast
		* Southwest"
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/teams"
    querystring = {}
    if name:
        querystring['name'] = name
    if league:
        querystring['league'] = league
    if is_id:
        querystring['id'] = is_id
    if code:
        querystring['code'] = code
    if division:
        querystring['division'] = division
    if search:
        querystring['search'] = search
    if conference:
        querystring['conference'] = conference
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_per_conference(search: str=None, division: str=None, conference: str='East', league: str=None, is_id: int=None, code: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about teams.
		
		The team `id` are **unique** in the API and teams keep it among all seasons.
		
		**Available conferences**
		* East
		* West
		
		**Available divisions**
		* Atlantic
		* Central
		* Northwest
		* Pacific
		* Southeast
		* Southwest"
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/teams"
    querystring = {}
    if search:
        querystring['search'] = search
    if division:
        querystring['division'] = division
    if conference:
        querystring['conference'] = conference
    if league:
        querystring['league'] = league
    if is_id:
        querystring['id'] = is_id
    if code:
        querystring['code'] = code
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_per_date(team: int=None, league: str=None, date: str='2022-02-12', season: int=None, is_id: int=None, h2h: str=None, live: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of games according to the parameters.
		
		**Available status**
		* 1 : Not Started
		* 2 : live
		* 3 : Finished
		
		> This endpoint requires at least one parameter."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/games"
    querystring = {}
    if team:
        querystring['team'] = team
    if league:
        querystring['league'] = league
    if date:
        querystring['date'] = date
    if season:
        querystring['season'] = season
    if is_id:
        querystring['id'] = is_id
    if h2h:
        querystring['h2h'] = h2h
    if live:
        querystring['live'] = live
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_in_live(league: str=None, season: int=None, live: str='all', team: int=None, is_id: int=None, date: str=None, h2h: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of games according to the parameters.
		
		**Available status**
		* 1 : Not Started
		* 2 : live
		* 3 : Finished
		
		> This endpoint requires at least one parameter."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/games"
    querystring = {}
    if league:
        querystring['league'] = league
    if season:
        querystring['season'] = season
    if live:
        querystring['live'] = live
    if team:
        querystring['team'] = team
    if is_id:
        querystring['id'] = is_id
    if date:
        querystring['date'] = date
    if h2h:
        querystring['h2h'] = h2h
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings(team: int=None, division: str=None, season: int=2021, league: str='standard', conference: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the standings for a league & season.
		
		**Available conferences**
		* East
		* West
		
		**Available divisions**
		* Atlantic
		* Central
		* Northwest
		* Pacific
		* Southeast
		* Southwest
		
		> Parameters `league` & `season` are required."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/standings"
    querystring = {}
    if team:
        querystring['team'] = team
    if division:
        querystring['division'] = division
    if season:
        querystring['season'] = season
    if league:
        querystring['league'] = league
    if conference:
        querystring['conference'] = conference
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_statistics(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics of the teams that played a game."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/games/statistics"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games(league: str=None, live: str=None, season: int=None, is_id: int=None, date: str='2022-02-12', h2h: str=None, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of games according to the parameters.
		
		**Available status**
		* 1 : Not Started
		* 2 : live
		* 3 : Finished
		
		> This endpoint requires at least one parameter."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/games"
    querystring = {}
    if league:
        querystring['league'] = league
    if live:
        querystring['live'] = live
    if season:
        querystring['season'] = season
    if is_id:
        querystring['id'] = is_id
    if date:
        querystring['date'] = date
    if h2h:
        querystring['h2h'] = h2h
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_statistics(is_id: int, season: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the overall statistics of a team for a given `season`."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/teams/statistics"
    querystring = {'id': is_id, 'season': season, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(code: str=None, search: str=None, league: str=None, conference: str=None, division: str=None, name: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about teams.
		
		The team `id` are **unique** in the API and teams keep it among all seasons.
		
		**Available conferences**
		* East
		* West
		
		**Available divisions**
		* Atlantic
		* Central
		* Northwest
		* Pacific
		* Southeast
		* Southwest"
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/teams"
    querystring = {}
    if code:
        querystring['code'] = code
    if search:
        querystring['search'] = search
    if league:
        querystring['league'] = league
    if conference:
        querystring['conference'] = conference
    if division:
        querystring['division'] = division
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues.
		
		All `leagues` can be used in other endpoints as filters.
		
		**Available leagues**
		* Africa
		* Orlando
		* Sacramento
		* Standard
		* Utah
		* Vegas
		
		> This endpoint does not require any parameters."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/leagues"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All `seasons` can be used in other endpoints as filters. Seasons are only **4-digit keys** like `YYYY`.
		
		> This endpoint does not require any parameters."
    
    """
    url = f"https://api-nba-v1.p.rapidapi.com/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

