import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def countries(name: str=None, is_id: int=None, code: str=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available countries | The {country} and {code} keys can be used in other endpoints as filters"
    name: Fails if field contains anything other than alpha-numeric characters, numbers or space
        is_id: Fails if field contains anything other than an integer
        code: 2 characters | Fails if field has anything other than alphabetic characters | Ex : FR, GB, IT...
        search: 3 characters minimum | Fails if field has anything other than alphabetic characters
        
    """
    url = f"https://api-basketball.p.rapidapi.com/countries"
    querystring = {}
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    if code:
        querystring['code'] = code
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available seasons | All {season} can be used in other endpoints as filters"
    
    """
    url = f"https://api-basketball.p.rapidapi.com/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues(country: str=None, is_id: int=None, type: str=None, season: str=None, name: str=None, code: str=None, search: str=None, country_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available leagues | The league {id} are unique in the API and leagues keep it across all seasons"
    country: Fails if field contains anything other than alpha-numeric characters, numbers or space
        is_id: Fails if field contains anything other than an integer
        type: Fails if field is not within a predetermined list : [league,cup]
        season: Fails if field contains anything other than a valid season [YYYY] or [YYYY-YYYY]
        name: Fails if field contains anything other than alpha-numeric characters, numbers or space
        code: 2 characters | Fails if field has anything other than alphabetic characters | Ex : FR, GB, IT...
        search: 3 characters minimum | Fails if field has anything other than alphabetic characters
        country_id: Fails if field contains anything other than an integer
        
    """
    url = f"https://api-basketball.p.rapidapi.com/leagues"
    querystring = {}
    if country:
        querystring['country'] = country
    if is_id:
        querystring['id'] = is_id
    if type:
        querystring['type'] = type
    if season:
        querystring['season'] = season
    if name:
        querystring['name'] = name
    if code:
        querystring['code'] = code
    if search:
        querystring['search'] = search
    if country_id:
        querystring['country_id'] = country_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def head_2_head(h2h: str, league: int=None, season: str=None, timezone: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all head to head between two teams"
    h2h: Fails if field does not match the regular expression : [id-id]
        league: Fails if field contains anything other than an integer
        season: Fails if field contains anything other than a valid season [YYYY] or [YYYY-YYYY]
        timezone: Fails if field is not a result of the endpoint timezone
        
    """
    url = f"https://api-basketball.p.rapidapi.com/games"
    querystring = {'h2h': h2h, }
    if league:
        querystring['league'] = league
    if season:
        querystring['season'] = season
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bookmakers(search: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get all available bookmakers | All bookmakers {id} can be used in endpoint [odds] as filters"
    search: 3 characters minimum | Fails if field has anything other than alphabetic characters
        is_id: Fails if field contains anything other than an integer
        
    """
    url = f"https://api-basketball.p.rapidapi.com/bookmakers"
    querystring = {}
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bets(is_id: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get all available bets labels | All bets {id} can be used in endpoint [odds] as filters"
    is_id: Fails if field contains anything other than an integer
        search: 3 characters minimum | Fails if field has anything other than alphabetic characters
        
    """
    url = f"https://api-basketball.p.rapidapi.com/bets"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_statistics(season: str, league: int, team: int, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team statistics"
    season: Fails if field contains anything other than a valid season [YYYY] or [YYYY-YYYY]
        league: Fails if field contains anything other than an integer
        team: Fails if field contains anything other than an integer
        date: Fails if field does not contain a valid date : [YYYY-MM-DD]
        
    """
    url = f"https://api-basketball.p.rapidapi.com/statistics"
    querystring = {'season': season, 'league': league, 'team': team, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_informations(name: str=None, league: int=12, is_id: int=None, season: str='2019-2020', search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available teams | The team {id} are unique in the API and teams keep it among all the leagues/cups in which they participate"
    name: Fails if field contains anything other than alpha-numeric characters, numbers, dashes or space
        league: Fails if field contains anything other than an integer
        is_id: Fails if field contains anything other than an integer
        season: Fails if field contains anything other than a valid season [YYYY] or [YYYY-YYYY]
        search: 3 characters minimum | Fails if field has anything other than alphabetic characters
        
    """
    url = f"https://api-basketball.p.rapidapi.com/teams"
    querystring = {}
    if name:
        querystring['name'] = name
    if league:
        querystring['league'] = league
    if is_id:
        querystring['id'] = is_id
    if season:
        querystring['season'] = season
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games(timezone: str=None, season: str=None, is_id: int=None, league: int=None, date: str='2019-11-26', team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available games"
    timezone: Fails if field is not a result of the endpoint timezone
        season: Fails if field contains anything other than a valid season [YYYY] or [YYYY-YYYY]
        is_id: Fails if field contains anything other than an integer
        league: Fails if field contains anything other than an integer
        date: Fails if field does not contain a valid date : [YYYY-MM-DD]
        team: Fails if field contains anything other than an integer
        
    """
    url = f"https://api-basketball.p.rapidapi.com/games"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    if season:
        querystring['season'] = season
    if is_id:
        querystring['id'] = is_id
    if league:
        querystring['league'] = league
    if date:
        querystring['date'] = date
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timezone(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available timezone to be used in the [games] endpoint"
    
    """
    url = f"https://api-basketball.p.rapidapi.com/timezone"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds(league: int=None, season: str=None, game: int=1912, bookmaker: int=None, bet: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get all available odds"
    league: Fails if field contains anything other than an integer
        season: Fails if field contains anything other than a valid season [YYYY] or [YYYY-YYYY]
        game: Fails if field contains anything other than an integer
        bookmaker: Fails if field contains anything other than an integer
        bet: Fails if field contains anything other than an integer
        
    """
    url = f"https://api-basketball.p.rapidapi.com/odds"
    querystring = {}
    if league:
        querystring['league'] = league
    if season:
        querystring['season'] = season
    if game:
        querystring['game'] = game
    if bookmaker:
        querystring['bookmaker'] = bookmaker
    if bet:
        querystring['bet'] = bet
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_bets(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get all available bets labels | All bets {id} can be used in endpoint [odds] as filters"
    search: 3 characters minimum | Fails if field has anything other than alphabetic characters
        
    """
    url = f"https://api-basketball.p.rapidapi.com/bets"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_bookmakers(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get all available bookmakers | All bookmakers {id} can be used in endpoint [odds] as filters"
    search: 3 characters minimum | Fails if field has anything other than alphabetic characters
        
    """
    url = f"https://api-basketball.p.rapidapi.com/bookmakers"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_countries(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available countries | The {country} and {code} keys can be used in other endpoints as filters"
    search: 3 characters minimum | Fails if field has anything other than alphabetic characters
        
    """
    url = f"https://api-basketball.p.rapidapi.com/countries"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_leagues(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available leagues | The league {id} are unique in the API and leagues keep it across all seasons"
    search: 3 characters minimum | Fails if field has anything other than alphabetic characters
        
    """
    url = f"https://api-basketball.p.rapidapi.com/leagues"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_teams(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available teams | The team {id} are unique in the API and teams keep it among all the leagues/cups in which they participate"
    search: 3 characters minimum | Fails if field has anything other than alphabetic characters
        
    """
    url = f"https://api-basketball.p.rapidapi.com/teams"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings(league: int, season: str, stage: str=None, group: str=None, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available standings | Return a table of one or more rankings according to the league / cup"
    league: Fails if field contains anything other than an integer
        season: Fails if field contains anything other than a valid season [YYYY] or [YYYY-YYYY]
        team: Fails if field contains anything other than an integer
        
    """
    url = f"https://api-basketball.p.rapidapi.com/standings"
    querystring = {'league': league, 'season': season, }
    if stage:
        querystring['stage'] = stage
    if group:
        querystring['group'] = group
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stages(league: int, season: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available stages for standings"
    league: Fails if field contains anything other than an integer
        season: Fails if field contains anything other than a valid season [YYYY] or [YYYY-YYYY]
        
    """
    url = f"https://api-basketball.p.rapidapi.com/standings/stages"
    querystring = {'league': league, 'season': season, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def groups(season: str, league: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available groups for standings"
    season: Fails if field contains anything other than a valid season [YYYY] or [YYYY-YYYY]
        league: Fails if field contains anything other than an integer
        
    """
    url = f"https://api-basketball.p.rapidapi.com/standings/groups"
    querystring = {'season': season, 'league': league, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-basketball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

