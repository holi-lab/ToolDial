import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def games_h2h(h2h: str, date: str=None, league: int=None, season: int=None, timezone: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get heads to heads between two teams."
    h2h: The ids of the teams
        date: A valid date
        league: The id of the league
        season: The season of the league
        timezone: A valid timezone
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/games/h2h"
    querystring = {'h2h': h2h, }
    if date:
        querystring['date'] = date
    if league:
        querystring['league'] = league
    if season:
        querystring['season'] = season
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games(timezone: str=None, season: int=None, league: int=None, team: int=None, is_id: int=None, date: str='2021-10-03', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all requests to games you can add the query parameter `timezone` to your request in order to retrieve the list of games in the time zone of your choice like *“Europe/London“*
		
		To know the list of available time zones you have to use the endpoint timezone.
		
		**Available status**
		
		        * NS : Not Started
		        * S1 : Set 1 (In Play)
		        * S2 : Set 2 (In Play)
		        * S3 : Set 3 (In Play)
		        * S4 : Set 4 (In Play)
		        * S5 : Set 5 (In Play)
		        * AW : Awarded
		        * POST : Postponed
		        * CANC : Cancelled
		        * INTR : Interrupted
		        * ABD : Abandoned
		        * FT : Finished (Game Finished)
		
		> Games are updated every 15 seconds
		
		> **This endpoint requires at least one parameter.**"
    timezone: A valid timezone
        season: The season of the league
        league: The id of the league
        team: The id of the team
        is_id: The id of the game
        date: A valid date
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/games"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    if season:
        querystring['season'] = season
    if league:
        querystring['league'] = league
    if team:
        querystring['team'] = team
    if is_id:
        querystring['id'] = is_id
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_bets(search: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bets.
		
		All bets `id` can be used in endpoint odds as filters"
    search: The name of the bet
        is_id: The id of the bet
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/odds/bets"
    querystring = {}
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_bookmakers(search: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bookmakers.
		
		All bookmakers `id` can be used in endpoint odds as filters"
    search: The name of the bet
        is_id: The id of the bookmakers
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/odds/bookmakers"
    querystring = {}
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_teams(name: str=None, country_id: int=None, season: int=None, country: str=None, league: int=None, search: str='Banfield', is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about teams.
		
		The team `id` are **unique** in the API and teams keep it among all the leagues/cups in which they participate.
		
		> **This endpoint requires at least one parameter.**"
    name: The name of the team
        country_id: The id of the country
        season: The season of the league
        country: The name of the country
        league: The id of the league
        search: >= 3 characters
        is_id: The id of the team
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/teams"
    querystring = {}
    if name:
        querystring['name'] = name
    if country_id:
        querystring['country_id'] = country_id
    if season:
        querystring['season'] = season
    if country:
        querystring['country'] = country
    if league:
        querystring['league'] = league
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_leagues(country: str=None, search: str=None, type: str=None, is_id: int=None, name: str=None, country_id: int=None, season: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues and cups.
		
		The league `id` are **unique** in the API and leagues keep it across all `seasons`
		
		> Most of the parameters of this endpoint can be used together."
    country: The name of the country
        search: >= 3 characters
        type:  Enum: \\\"league\\\" \\\"cup\\\"

The type of the league

        is_id: The id of the league
        name: The name of the league
        country_id: The id of the country
        season: The season of the league
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/leagues"
    querystring = {}
    if country:
        querystring['country'] = country
    if search:
        querystring['search'] = search
    if type:
        querystring['type'] = type
    if is_id:
        querystring['id'] = is_id
    if name:
        querystring['name'] = name
    if country_id:
        querystring['country_id'] = country_id
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_countries(name: str=None, code: str=None, search: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available countries.
		
		The `name`, `id` and `code` fields can be used in other endpoints as filters.
		
		> All the parameters of this endpoint can be used together."
    name: The name of the country
        code: The code of the country
        search: >= 3 characters
        is_id: The id of the country
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/countries"
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
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds_bookmakers(search: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bookmakers.
		
		All bookmakers `id` can be used in endpoint odds as filters"
    search: The name of the bet
        is_id: The id of the bookmakers
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/odds/bookmakers"
    querystring = {}
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds_bets(is_id: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bets.
		
		All bets `id` can be used in endpoint odds as filters"
    is_id: The id of the bet
        search: The name of the bet
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/odds/bets"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds(league: int=None, bet: int=None, bookmaker: int=None, season: int=None, game: int=3546, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get odds from games or leagues.
		
		We provide pre-match odds between 1 and 7 days before the game.
		
		We keep a 7-day history *(The availability of odds may vary according to the leagues, seasons, games and bookmakers)*
		
		> Odds are updated once a day"
    league: The id of the league
        bet: The id of the bet
        bookmaker: The id of the bookmaker
        season: The season of the league
        game: The id of the game
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/odds"
    querystring = {}
    if league:
        querystring['league'] = league
    if bet:
        querystring['bet'] = bet
    if bookmaker:
        querystring['bookmaker'] = bookmaker
    if season:
        querystring['season'] = season
    if game:
        querystring['game'] = game
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings_groups(league: int, season: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available groups for a league to be used in the standings endpoint."
    league: The id of the league
        season: The season of the league
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/standings/groups"
    querystring = {'league': league, 'season': season, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings_stages(season: int, league: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available stages for a league to be used in the standings endpoint."
    season: The season of the league
        league: The id of the league
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/standings/stages"
    querystring = {'season': season, 'league': league, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings(season: int, league: int, stage: str=None, team: int=None, group: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the standings for a league.
		
		        Return a table of one or more rankings according to the league / cup. Some competitions have several rankings in a year, regular season, pre season etc…
		
		To know the list of available stages or groups you have to use the endpoint standings/stages or standings/groups
		
		> Standings are updated every hours"
    season: The season of the league
        league: The id of the league
        stage: A valid stage
        team: The id of the team
        group: A valid group
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/standings"
    querystring = {'season': season, 'league': league, }
    if stage:
        querystring['stage'] = stage
    if team:
        querystring['team'] = team
    if group:
        querystring['group'] = group
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_statistics(league: int=2, team: int=7, date: str=None, season: int=2020, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the statistics for a team according to a league and season"
    league: The id of the league
        team: The id of the team
        date: A date limit
        season: The season of the league
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/teams/statistics"
    querystring = {}
    if league:
        querystring['league'] = league
    if team:
        querystring['team'] = team
    if date:
        querystring['date'] = date
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(country: str=None, is_id: int=None, league: int=2, country_id: int=None, season: int=2020, name: str=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about teams.
		
		The team `id` are **unique** in the API and teams keep it among all the leagues/cups in which they participate.
		
		> **This endpoint requires at least one parameter.**"
    country: The name of the country
        is_id: The id of the team
        league: The id of the league
        country_id: The id of the country
        season: The season of the league
        name: The name of the team
        search: >= 3 characters
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/teams"
    querystring = {}
    if country:
        querystring['country'] = country
    if is_id:
        querystring['id'] = is_id
    if league:
        querystring['league'] = league
    if country_id:
        querystring['country_id'] = country_id
    if season:
        querystring['season'] = season
    if name:
        querystring['name'] = name
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues(search: str=None, season: int=None, type: str=None, country_id: int=None, name: str=None, is_id: int=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues and cups.
		
		The league `id` are **unique** in the API and leagues keep it across all `seasons`
		
		> Most of the parameters of this endpoint can be used together."
    search: >= 3 characters
        season: The season of the league
        type:  Enum: \"league\" \"cup\"

The type of the league

        country_id: The id of the country
        name: The name of the league
        is_id: The id of the league
        country: The name of the country
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/leagues"
    querystring = {}
    if search:
        querystring['search'] = search
    if season:
        querystring['season'] = season
    if type:
        querystring['type'] = type
    if country_id:
        querystring['country_id'] = country_id
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(search: str=None, is_id: int=None, name: str=None, code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available countries.
		
		The `name`, `id` and `code` fields can be used in other endpoints as filters.
		
		> All the parameters of this endpoint can be used together."
    search: >= 3 characters
        is_id: The id of the country
        name: The name of the country
        code: The code of the country
        
    """
    url = f"https://api-volleyball.p.rapidapi.com/countries"
    querystring = {}
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    if name:
        querystring['name'] = name
    if code:
        querystring['code'] = code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All seasons are only **4-digit keys**, so for a league whose season is `2018-2019` the season in the API will be `2018`.
		
		All `seasons` can be used in other endpoints as filters.
		
		> This endpoint does not require any parameters."
    
    """
    url = f"https://api-volleyball.p.rapidapi.com/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timezone(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available timezone to be used in the games endpoint.
		
		> This endpoint does not require any parameters."
    
    """
    url = f"https://api-volleyball.p.rapidapi.com/timezone"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-volleyball.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

