import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def timezone(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "timezone"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/timezone"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bets(is_id: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bets"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/odds/bets"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bookmakers(is_id: str=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "bookmakers"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/odds/bookmakers"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_statistics(date: str=None, league: str='1', season: str='2019', team: str='5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "teams statistics"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/teams/statistics"
    querystring = {}
    if date:
        querystring['date'] = date
    if league:
        querystring['league'] = league
    if season:
        querystring['season'] = season
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(name: str=None, code: str=None, search: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "countries"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/countries"
    querystring = {}
    if name:
        querystring['name'] = name
    if code:
        querystring['code'] = code
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "seasons"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_bets(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search bets"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/odds/bets"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_bookmakers(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search bookmakers"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/odds/bookmakers"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_countries(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search countries"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/countries"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_leagues(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search leagues"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/leagues"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_teams(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search teams"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/teams"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stages(season: int, league: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "stages"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/standings/stages"
    querystring = {'season': season, 'league': league, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def groups(season: int, league: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "groups"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/standings/groups"
    querystring = {'season': season, 'league': league, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings(group: str=None, league: int=1, team: int=None, stage: str=None, season: int=2020, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "standings"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/standings"
    querystring = {}
    if group:
        querystring['group'] = group
    if league:
        querystring['league'] = league
    if team:
        querystring['team'] = team
    if stage:
        querystring['stage'] = stage
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues(country_id: int=None, search: str=None, type: str=None, country: str=None, season: int=None, is_id: int=None, name: str='mlb', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "leagues"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/leagues"
    querystring = {}
    if country_id:
        querystring['country_id'] = country_id
    if search:
        querystring['search'] = search
    if type:
        querystring['type'] = type
    if country:
        querystring['country'] = country
    if season:
        querystring['season'] = season
    if is_id:
        querystring['id'] = is_id
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games(league: int=None, season: int=None, team: int=None, is_id: int=1, date: str=None, timezone: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "games"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/games"
    querystring = {}
    if league:
        querystring['league'] = league
    if season:
        querystring['season'] = season
    if team:
        querystring['team'] = team
    if is_id:
        querystring['id'] = is_id
    if date:
        querystring['date'] = date
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def head_to_head(h2h: str, season: int=None, timezone: str=None, league: int=None, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "games head to head"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/games/h2h"
    querystring = {'h2h': h2h, }
    if season:
        querystring['season'] = season
    if timezone:
        querystring['timezone'] = timezone
    if league:
        querystring['league'] = league
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds(game: int=5, season: int=None, league: int=None, bookmaker: int=None, bet: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "odds"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/odds"
    querystring = {}
    if game:
        querystring['game'] = game
    if season:
        querystring['season'] = season
    if league:
        querystring['league'] = league
    if bookmaker:
        querystring['bookmaker'] = bookmaker
    if bet:
        querystring['bet'] = bet
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(league: int=1, country: str=None, search: str=None, name: str=None, is_id: int=None, season: int=2020, country_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "teams"
    
    """
    url = f"https://api-baseball.p.rapidapi.com/teams"
    querystring = {}
    if league:
        querystring['league'] = league
    if country:
        querystring['country'] = country
    if search:
        querystring['search'] = search
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    if season:
        querystring['season'] = season
    if country_id:
        querystring['country_id'] = country_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-baseball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

