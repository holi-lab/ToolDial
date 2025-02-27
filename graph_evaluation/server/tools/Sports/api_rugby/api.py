import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def seasons(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All seasons are only **4-digit keys**, so for a league whose season is `2018-2019` the season in the API will be `2018`.
		
		All `seasons` can be used in other endpoints as filters.
		
		> This endpoint does not require any parameters."
    
    """
    url = f"https://api-rugby.p.rapidapi.com/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings(league: int, season: int, group: str=None, stage: str=None, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the standings for a league.
		
		Return a table of one or more rankings according to the league / cup. Some competitions have several rankings in a year, regular season, pre season etc…
		
		To know the list of available stages or groups you have to use the endpoint standings/stages or standings/groups
		
		> Standings are updated every hours"
    league: The id of the league
        season: The season of the league
        group: A valid group
        stage: A valid stage
        team: The id of the team
        
    """
    url = f"https://api-rugby.p.rapidapi.com/standings"
    querystring = {'league': league, 'season': season, }
    if group:
        querystring['group'] = group
    if stage:
        querystring['stage'] = stage
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues(country_id: int=None, is_id: int=None, name: str=None, season: int=None, country: str=None, search: str=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues and cups.
		
		The league `id` are **unique** in the API and leagues keep it across all `seasons`
		
		> Most of the parameters of this endpoint can be used together."
    country_id: The id of the country
        is_id: The id of the league
        name: The name of the league
        season: The season of the league
        country: The name of the country
        type: The type of the league [league, cup]
        
    """
    url = f"https://api-rugby.p.rapidapi.com/leagues"
    querystring = {}
    if country_id:
        querystring['country_id'] = country_id
    if is_id:
        querystring['id'] = is_id
    if name:
        querystring['name'] = name
    if season:
        querystring['season'] = season
    if country:
        querystring['country'] = country
    if search:
        querystring['search'] = search
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(search: str=None, code: str=None, name: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available countries.
		
		The `name`, `id` and `code` fields can be used in other endpoints as filters.
		
		> All the parameters of this endpoint can be used together."
    code: The code of the country
        name: The name of the country
        is_id: The id of the country
        
    """
    url = f"https://api-rugby.p.rapidapi.com/countries"
    querystring = {}
    if search:
        querystring['search'] = search
    if code:
        querystring['code'] = code
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bookmakers(is_id: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bookmakers.
		
		All bookmakers `id` can be used in endpoint odds as filters."
    is_id: The id of the bookmaker
        search: The name of the bookmaker
        
    """
    url = f"https://api-rugby.p.rapidapi.com/odds/bookmakers"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bets(is_id: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bets.
		
		All bets `id` can be used in endpoint odds as filters"
    is_id: The id of the bet
        search: The name of the bet
        
    """
    url = f"https://api-rugby.p.rapidapi.com/odds/bets"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds(bet: int=None, season: int=None, league: int=None, bookmaker: int=None, game: int=1107, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get odds from games or leagues.
		
		We provide pre-match odds between 1 and 7 days before the game.
		
		We keep a 7-day history *(The availability of odds may vary according to the leagues, seasons, games and bookmakers)*
		
		> Odds are updated once a day"
    bet: The id of the bet
        season: The season of the league
        league: The id of the league
        bookmaker: The id of the bookmaker
        game: The id of the game
        
    """
    url = f"https://api-rugby.p.rapidapi.com/odds"
    querystring = {}
    if bet:
        querystring['bet'] = bet
    if season:
        querystring['season'] = season
    if league:
        querystring['league'] = league
    if bookmaker:
        querystring['bookmaker'] = bookmaker
    if game:
        querystring['game'] = game
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
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
    url = f"https://api-rugby.p.rapidapi.com/timezone"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games(league: int=None, timezone: str=None, season: int=None, date: str=None, team: int=None, is_id: int=8279, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all requests to games you can add the query parameter `timezone` to your request in order to retrieve the list of games in the time zone of your choice like *“Europe/London“*
		
		To know the list of available time zones you have to use the endpoint timezone.
		
		**Available status**
		
		       
		
		> Games are updated every 15 seconds
		
		> **This endpoint requires at least one parameter.**"
    league: The id of the league
        timezone: A valid timezone
        season: The season of the league
        date: A valid date
        team: The id of the team
        is_id: The id of the game
        
    """
    url = f"https://api-rugby.p.rapidapi.com/games"
    querystring = {}
    if league:
        querystring['league'] = league
    if timezone:
        querystring['timezone'] = timezone
    if season:
        querystring['season'] = season
    if date:
        querystring['date'] = date
    if team:
        querystring['team'] = team
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_statistics(league: int, team: int, season: int, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the statistics of a team in relation to a given competition and season.
		
		It is possible to add the `date` parameter to calculate statistics from the beginning of the season to the given date. By default the API returns the statistics of all games played by the team for the competition and the season."
    league: The id of the league
        team: The id of the team
        season: The season of the league
        date: A date limit
        
    """
    url = f"https://api-rugby.p.rapidapi.com/teams/statistics"
    querystring = {'league': league, 'team': team, 'season': season, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings_stages(league: int, season: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available stages for a league to be used in the standings endpoint."
    league: The id of the league
        season: The season of the league
        
    """
    url = f"https://api-rugby.p.rapidapi.com/standings/stages"
    querystring = {'league': league, 'season': season, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
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
    url = f"https://api-rugby.p.rapidapi.com/standings/groups"
    querystring = {'league': league, 'season': season, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def head_to_heads(date: str=None, timezone: str=None, league: int=None, season: int=None, h2h: str='368-367', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get heads to heads between two teams."
    date: A valid date
        timezone: A valid timezone
        league: The id of the league
        season: The season of the league
        h2h: The ids of the teams
        
    """
    url = f"https://api-rugby.p.rapidapi.com/games/h2h"
    querystring = {}
    if date:
        querystring['date'] = date
    if timezone:
        querystring['timezone'] = timezone
    if league:
        querystring['league'] = league
    if season:
        querystring['season'] = season
    if h2h:
        querystring['h2h'] = h2h
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_leagues(search: str='top 14', name: str=None, type: str=None, country_id: int=None, is_id: int=None, country: str=None, season: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues and cups.
		
		The league `id` are **unique** in the API and leagues keep it across all `seasons`
		
		> Most of the parameters of this endpoint can be used together."
    name: The name of the league
        type: The type of the league [league, cup]
        country_id: The id of the country
        is_id: The id of the league
        country: The name of the country
        season: The season of the league
        
    """
    url = f"https://api-rugby.p.rapidapi.com/leagues"
    querystring = {}
    if search:
        querystring['search'] = search
    if name:
        querystring['name'] = name
    if type:
        querystring['type'] = type
    if country_id:
        querystring['country_id'] = country_id
    if is_id:
        querystring['id'] = is_id
    if country:
        querystring['country'] = country
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_countries(search: str='france', code: str=None, name: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available countries.
		
		The `name`, `id` and `code` fields can be used in other endpoints as filters.
		
		> All the parameters of this endpoint can be used together."
    code: The code of the country
        name: The name of the country
        is_id: The id of the country
        
    """
    url = f"https://api-rugby.p.rapidapi.com/countries"
    querystring = {}
    if search:
        querystring['search'] = search
    if code:
        querystring['code'] = code
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_bookmakers(search: str='Bet365', is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bookmakers.
		
		All bookmakers `id` can be used in endpoint odds as filters."
    search: The name of the bookmaker
        is_id: The id of the bookmaker
        
    """
    url = f"https://api-rugby.p.rapidapi.com/odds/bookmakers"
    querystring = {}
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_bets(is_id: int=None, search: str='Home', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bets.
		
		All bets `id` can be used in endpoint odds as filters"
    is_id: The id of the bet
        search: The name of the bet
        
    """
    url = f"https://api-rugby.p.rapidapi.com/odds/bets"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(country_id: int=None, country: str=None, is_id: int=119, season: int=None, name: str=None, search: str=None, league: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about teams.
		
		The team `id` are **unique** in the API and teams keep it among all the leagues/cups in which they participate.
		
		> **This endpoint requires at least one parameter.**"
    country_id: The id of the country
        country: The name of the country
        is_id: The id of the team
        season: The season of the league
        name: The name of the team
        league: The id of the league
        
    """
    url = f"https://api-rugby.p.rapidapi.com/teams"
    querystring = {}
    if country_id:
        querystring['country_id'] = country_id
    if country:
        querystring['country'] = country
    if is_id:
        querystring['id'] = is_id
    if season:
        querystring['season'] = season
    if name:
        querystring['name'] = name
    if search:
        querystring['search'] = search
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_teams(country: str=None, is_id: int=None, league: int=None, season: int=None, name: str=None, search: str='Rouen', country_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about teams.
		
		The team `id` are **unique** in the API and teams keep it among all the leagues/cups in which they participate.
		
		> **This endpoint requires at least one parameter.**"
    country: The name of the country
        is_id: The id of the team
        league: The id of the league
        season: The season of the league
        name: The name of the team
        country_id: The id of the country
        
    """
    url = f"https://api-rugby.p.rapidapi.com/teams"
    querystring = {}
    if country:
        querystring['country'] = country
    if is_id:
        querystring['id'] = is_id
    if league:
        querystring['league'] = league
    if season:
        querystring['season'] = season
    if name:
        querystring['name'] = name
    if search:
        querystring['search'] = search
    if country_id:
        querystring['country_id'] = country_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-rugby.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

