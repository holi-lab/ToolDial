import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_specific_team(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves a specific team"
    id: The ID of the team
        
    """
    url = f"https://free-nba.p.rapidapi.com/teams/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_games(page: str='0', per_page: str='25', team_ids: str=None, date: str=None, seasons: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves all games.  Seasons are represented by the year they began. For example, 2018 represents season 2018-2019."
    page: The page number, used for pagination.
        per_page: The number of results returned per call, used for pagination.
        team_ids: An array of team_ids
        date: An array of dates formatted in 'YYYY-MM-DD'
        seasons: An array of seasons
        
    """
    url = f"https://free-nba.p.rapidapi.com/games"
    querystring = {}
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    if team_ids:
        querystring['team_ids'] = team_ids
    if date:
        querystring['date'] = date
    if seasons:
        querystring['Seasons'] = seasons
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_specific_game(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves a specific game"
    id: ID of the game
        
    """
    url = f"https://free-nba.p.rapidapi.com/games/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_stats(seasons: str=None, page: str='0', per_page: str='25', player_ids: str=None, dates: str=None, game_ids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves all stats"
    seasons: An array of seasons
        page: The page number, used for pagination.
        per_page: The number of results returned per call, used for pagination.
        player_ids: An array of player_ids
        dates: An array of dates formatted in 'YYYY-MM-DD'
        game_ids: An array of game_ids
        
    """
    url = f"https://free-nba.p.rapidapi.com/stats"
    querystring = {}
    if seasons:
        querystring['seasons'] = seasons
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    if player_ids:
        querystring['player_ids'] = player_ids
    if dates:
        querystring['dates'] = dates
    if game_ids:
        querystring['game_ids'] = game_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_players(page: str='0', per_page: str='25', search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves all players from all seasons."
    page: The page number, used for pagination.
        per_page: The number of results returned per call, used for pagination.
        search: Used to filter players based on their name. For example, ?search=davis will return players that have 'davis' in their first or last name.
        
    """
    url = f"https://free-nba.p.rapidapi.com/players"
    querystring = {}
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_specific_player(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves a specific player."
    id: The ID of the player to retrieve
        
    """
    url = f"https://free-nba.p.rapidapi.com/players/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_teams(page: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint retrieves all teams for the current season."
    page: The page number, used for pagination.
        
    """
    url = f"https://free-nba.p.rapidapi.com/teams"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "free-nba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

