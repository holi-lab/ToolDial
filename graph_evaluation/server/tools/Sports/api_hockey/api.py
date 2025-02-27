import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def timezone(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available timezone to be used in the games endpoint.
		
		> This endpoint does not require any parameters."
    
    """
    url = f"https://api-hockey.p.rapidapi.com/timezone"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All seasons are only 4-digit keys, so for a league whose season is 2018-2019 the season in the API will be 2018.
		
		All seasons can be used in other endpoints as filters.
		
		> This endpoint does not require any parameters."
    
    """
    url = f"https://api-hockey.p.rapidapi.com/seasons/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(search: str=None, code: str=None, is_id: int=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available countries.
		
		The name and code fields can be used in other endpoints as filters.
		
		> All the parameters of this endpoint can be used together."
    code: The code of the country
        is_id: The id of the country
        name: The name of the country
        
    """
    url = f"https://api-hockey.p.rapidapi.com/countries/"
    querystring = {}
    if search:
        querystring['search'] = search
    if code:
        querystring['code'] = code
    if is_id:
        querystring['id'] = is_id
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
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
    url = f"https://api-hockey.p.rapidapi.com/countries/"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues(season: int=None, is_id: int=None, country: str=None, search: str=None, country_id: int=None, type: str=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues and cups.
		
		The league id are unique in the API and leagues keep it across all seasons
		
		> Most of the parameters of this endpoint can be used together."
    season: The season of the league
        is_id: The id of the league
        country: The name of the country
        country_id: The id of the country
        type: The type of the league : league or cup
        name: The name of the league
        
    """
    url = f"https://api-hockey.p.rapidapi.com/leagues/"
    querystring = {}
    if season:
        querystring['season'] = season
    if is_id:
        querystring['id'] = is_id
    if country:
        querystring['country'] = country
    if search:
        querystring['search'] = search
    if country_id:
        querystring['country_id'] = country_id
    if type:
        querystring['type'] = type
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_leagues(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Leagues"
    
    """
    url = f"https://api-hockey.p.rapidapi.com/leagues/"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_statistics(team: int, league: int, season: int, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Teams statistics for a league and a season"
    team: The id of the team
        league: The id of the league
        season: The season of the league
        date: A date limit
        
    """
    url = f"https://api-hockey.p.rapidapi.com/teams/statistics/"
    querystring = {'team': team, 'league': league, 'season': season, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings(league: int, season: int, stage: str=None, group: str=None, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the standings for a league.
		
		Return a table of one or more rankings according to the league / cup. Some competitions have several rankings in a year, regular season, pre season etc…
		
		To know the list of available stages or grou^ you have to use the endpoint standings/stages or standings/groups
		
		> Standings are updated every hours"
    league: The id of the league
        season: The season of the league
        stage: A valid stage
        group: A valid group
        team: The id of the team
        
    """
    url = f"https://api-hockey.p.rapidapi.com/standings/"
    querystring = {'league': league, 'season': season, }
    if stage:
        querystring['stage'] = stage
    if group:
        querystring['group'] = group
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stages(season: int, league: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available stages for a league to be used in the standings endpoint."
    season: The season of the league
        league: The id of the league
        
    """
    url = f"https://api-hockey.p.rapidapi.com/standings/stages"
    querystring = {'season': season, 'league': league, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def groups(league: int, season: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available groups for a league to be used in the standings endpoint."
    league: The id of the league
        season: The season of the league
        
    """
    url = f"https://api-hockey.p.rapidapi.com/standings/groups"
    querystring = {'league': league, 'season': season, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games(league: int=None, is_id: int=None, season: int=None, timezone: str=None, team: int=None, date: str='2020-10-02', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For all requests to games you can add the query parameter timezone to your request in order to retrieve the list of games in the time zone of your choice like “Europe/London“
		
		To know the list of available time zones you have to use the endpoint timezone
		
		> Games are updated every 15 seconds
		
		> This endpoint requires at least one parameter."
    league: The id of the league
        is_id: The id of the game
        season: The season of the league
        timezone: A valid timezone
        team: The id of the team
        date: A valid date
        
    """
    url = f"https://api-hockey.p.rapidapi.com/games/"
    querystring = {}
    if league:
        querystring['league'] = league
    if is_id:
        querystring['id'] = is_id
    if season:
        querystring['season'] = season
    if timezone:
        querystring['timezone'] = timezone
    if team:
        querystring['team'] = team
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def heads_2_heads(h2h: str, date: str=None, league: int=None, season: int=None, timezone: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get heads to heads between two teams."
    h2h: The ids of the teams : id-id
        date: A valid date
        league: The id of the league
        season: The season of the league
        timezone: A valid timezone
        
    """
    url = f"https://api-hockey.p.rapidapi.com/games/h2h/"
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
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events(game: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all events from one game"
    game: The id of the game
        
    """
    url = f"https://api-hockey.p.rapidapi.com/games/events"
    querystring = {'game': game, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
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
		
		All bets id can be used in endpoint odds as filters"
    is_id: The id of the bet
        
    """
    url = f"https://api-hockey.p.rapidapi.com/odds/bets"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_bets(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Bets"
    
    """
    url = f"https://api-hockey.p.rapidapi.com/odds/bets/"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
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
		
		All bookmakers id can be used in endpoint odds as filters."
    is_id: The id of the bookmaker
        
    """
    url = f"https://api-hockey.p.rapidapi.com/odds/bookmakers"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_bookmakers(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Bookmakers"
    
    """
    url = f"https://api-hockey.p.rapidapi.com/odds/bookmakers/"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_details(league: int=None, name: str=None, search: str=None, country_id: int=None, country: str=None, season: int=None, is_id: int=135, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about teams.
		
		The team id are unique in the API and teams keep it among all the leagues/cups in which they participate.
		
		> This endpoint requires at least one parameter."
    league: The id of the league
        name: The name of the team
        country_id: The id of the country
        country: The name of the country
        season: The season of the league
        is_id: The id of the team
        
    """
    url = f"https://api-hockey.p.rapidapi.com/teams/"
    querystring = {}
    if league:
        querystring['league'] = league
    if name:
        querystring['name'] = name
    if search:
        querystring['search'] = search
    if country_id:
        querystring['country_id'] = country_id
    if country:
        querystring['country'] = country
    if season:
        querystring['season'] = season
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_teams(search: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Teams"
    
    """
    url = f"https://api-hockey.p.rapidapi.com/teams/"
    querystring = {'search': search, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds(game: int=11590, bookmaker: int=None, season: int=None, bet: int=None, league: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get odds from games or leagues.
		
		We provide pre-match odds between 1 and 7 days before the game.
		
		We keep a 7-day history (The availability of odds may vary according to the leagues, seasons, games and bookmakers)
		
		> Odds are updated once a day"
    game: The id of the game
        bookmaker: The id of the bookmaker
        season: The season of the league
        bet: The id of the bet
        league: The id of the league
        
    """
    url = f"https://api-hockey.p.rapidapi.com/odds/"
    querystring = {}
    if game:
        querystring['game'] = game
    if bookmaker:
        querystring['bookmaker'] = bookmaker
    if season:
        querystring['season'] = season
    if bet:
        querystring['bet'] = bet
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-hockey.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

