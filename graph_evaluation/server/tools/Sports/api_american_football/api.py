import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bets(search: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the list of available bets for odds.
		
		All bets `id` can be used in endpoint `odds` as filters
		
		**Parameters:** You can call this endpoint without any parameters to get the complete list."
    search: The name of the bet
        is_id: The id of the bet
        
    """
    url = f"https://api-american-football.p.rapidapi.com/odds/bets"
    querystring = {}
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bookmakers(is_id: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the list of available bookmakers for odds.
		
		All bookmakers `id` can be used in endpoint `odds` as filters.
		
		**Parameters:** You can call this endpoint without any parameters to get the complete list."
    is_id: The id of the bookmaker
        search: The name of the bookmaker
        
    """
    url = f"https://api-american-football.p.rapidapi.com/odds/bookmakers"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds(game: int, bet: str=None, bookmaker: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the list of available odds for games.
		
		We provide pre-match odds between 1 and 7 days before the game.
		
		We keep a 7-day history *(The availability of odds may vary according to the games, seasons and bookmakers)*
		
		**Parameters:** This endpoint requires at least one of theses parameters : game.
		
		> Odds are updated four times a day"
    game: The id of the game
        bet: The id of the bet
        bookmaker: The id of the bookmaker
        
    """
    url = f"https://api-american-football.p.rapidapi.com/odds"
    querystring = {'game': game, }
    if bet:
        querystring['bet'] = bet
    if bookmaker:
        querystring['bookmaker'] = bookmaker
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings_divisions(league: int, season: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the list of available divisions for a competition to be used in the `standings` endpoint.
		
		**Parameters:** This endpoint requires at least two parameters `league` and `season`."
    league: The id of the league
        season: The season of the league
        
    """
    url = f"https://api-american-football.p.rapidapi.com/standings/divisions"
    querystring = {'league': league, 'season': season, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings_conferences(season: int, league: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the list of available conferences for a competition to be used in the `standings` endpoint.
		
		**Parameters:** This endpoint requires at least two parameters `league` and `season`."
    season: The season of the league
        league: The id of the league
        
    """
    url = f"https://api-american-football.p.rapidapi.com/standings/conferences"
    querystring = {'season': season, 'league': league, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings(season: int, league: int, division: str=None, conference: str=None, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the standings of a competition in relation to a given season.
		
		To know the list of available `conferences` or `divisions` you have to use the endpoint `standings/conferences` or `standings/divisions`.
		
		**Parameters:** This endpoint requires at least two parameters `league` and `season`.
		
		> This endpoint is updated every hour"
    season: The season of the league
        league: The id of the league
        division: A valid division
        conference: A valid conference
        team: The id of the team
        
    """
    url = f"https://api-american-football.p.rapidapi.com/standings"
    querystring = {'season': season, 'league': league, }
    if division:
        querystring['division'] = division
    if conference:
        querystring['conference'] = conference
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_players_statistics(is_id: int, group: str=None, team: int=22, player: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return players statistics from a game `id`.
		
		The statistics of the players are different depending on the positions they occupy in the formation, so they are grouped into different `groups`.
		
		**Available Groups**
		  * defensive
		  * fumbles
		  * interceptions
		  * kick_returns
		  * kicking
		  * passing
		  * punt_returns
		  * punting
		  * receiving
		  * rushing
		
		**Parameters:** This endpoint requires at least one parameter.
		
		> This endpoint is updated every 30 seconds"
    id: The id of the game
        group: A valid group
        team: The id of the team
        player: The id of the player
        
    """
    url = f"https://api-american-football.p.rapidapi.com/games/statistics/players"
    querystring = {'id': is_id, }
    if group:
        querystring['group'] = group
    if team:
        querystring['team'] = team
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_teams_statistics(is_id: int, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return team statistics from a game `id`.
		
		**Parameters:** This endpoint requires at least one parameter.
		
		> This endpoint is updated every 30 seconds"
    id: The id of the game
        team: The id of the team
        
    """
    url = f"https://api-american-football.p.rapidapi.com/games/statistics/teams"
    querystring = {'id': is_id, }
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_events(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the list of events for one game `id`.
		
		**Parameters:** This endpoint requires at least one parameter.
		
		> This endpoint is updated every 30 seconds"
    id: The id of the game
        
    """
    url = f"https://api-american-football.p.rapidapi.com/games/events"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games(h2h: str=None, season: int=None, timezone: str=None, team: int=None, is_id: int=None, league: int=None, live: str=None, date: str='2022-09-30', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the list of games according to the given parameters.
		
		For all requests to games you can add the query parameter `timezone` to your request in order to retrieve the list of games in the time zone of your choice like *“Europe/London“*. In case the timezone is not recognized, empty or is not part of the endpoint `timezone` list, the `UTC` value will be applied by default
		
		To know the list of available timezones you have to use the endpoint `timezone`.
		
		**Available Status**
		
		* NS : Not Started
		* Q1 : First Quarter (In Play)
		* Q2 : Second Quarter (In Play)
		* Q3 : Third Quarter (In Play)
		* Q4 : Fourth Quarter (In Play)
		* OT : Overtime (In Play)
		* HT : Halftime (In Play)
		* FT : Finished (Game Finished)
		* AOT : After Over Time (Game Finished)
		* CANC : Cancelled (Game cancelled and not rescheduled)
		* PST : Postponed (Game postponed and waiting for a new game date)
		
		**Parameters:** This endpoint requires at least one of these parameters `id`, `date`, `league`, `team`, `live`, `h2h`.
		
		> Games are updated every 30 seconds"
    h2h: Two teams Ids
        season: The season of the league
        timezone: A valid timezone
        team: The id of the team
        is_id: The id of the game
        league: The id of the league
        live: value : `all`
        date: A valid date
        
    """
    url = f"https://api-american-football.p.rapidapi.com/games"
    querystring = {}
    if h2h:
        querystring['h2h'] = h2h
    if season:
        querystring['season'] = season
    if timezone:
        querystring['timezone'] = timezone
    if team:
        querystring['team'] = team
    if is_id:
        querystring['id'] = is_id
    if league:
        querystring['league'] = league
    if live:
        querystring['live'] = live
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def injuries(team: int=1, player: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the list of injured players.
		
		There is no preserved history, only the currently injured players appear in this endpoint.
		
		**Parameters:** This endpoint requires at least one parameter `id` or `team`.
		
		> This endpoint is updated every hour"
    team: The id of the team
        player: The id of the player
        
    """
    url = f"https://api-american-football.p.rapidapi.com/injuries"
    querystring = {}
    if team:
        querystring['team'] = team
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_statistics(season: int, team: int=None, is_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics of a player for the whole season.
		
		The statistics of the players are different depending on the positions they occupy in the formation, so they are grouped into different `groups`.
		
		**Available Groups**
		  * Defense
		  * Kicking
		  * Passing
		  * Punting
		  * Receiving
		  * Returning
		  * Rushing
		  * Scoring
		
		> **Data for this endpoint start from 2022 season**
		
		**Parameters:** This endpoint requires at least two parameters `id` or `team` and `season`.
		
		> This endpoint is updated every day"
    season: The season
        team: The id of the team
        is_id: The id of the player
        
    """
    url = f"https://api-american-football.p.rapidapi.com/players/statistics"
    querystring = {'season': season, }
    if team:
        querystring['team'] = team
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players(name: str=None, search: str=None, team: int=None, is_id: int=1, season: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a set of data about the players.
		
		The players `id` are **unique** in the API and  keep it among all the competitions in which they participate.
		
		*Not all data is available for all players.*
		
		**Parameters:** This endpoint requires at least one parameter.
		
		> This endpoint is updated every day"
    name: The name of the player
        search: The name of the player
        team: The id of the team
        is_id: The id of the player
        season: The season
        
    """
    url = f"https://api-american-football.p.rapidapi.com/players"
    querystring = {}
    if name:
        querystring['name'] = name
    if search:
        querystring['search'] = search
    if team:
        querystring['team'] = team
    if is_id:
        querystring['id'] = is_id
    if season:
        querystring['season'] = season
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(code: str=None, name: str=None, is_id: int=1, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a set of data about the teams.
		
		The team `id` are **unique** in the API and teams keep it among all the competitions in which they participate.
		
		**Parameters:** This endpoint requires at least one parameter.
		
		> This endpoint is updated every day"
    code: The code of the team
        name: The name of the team
        is_id: The id of the team
        search: The name of the team
        
    """
    url = f"https://api-american-football.p.rapidapi.com/teams"
    querystring = {}
    if code:
        querystring['code'] = code
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues(is_id: int=None, season: int=None, current: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the list of all available competitions.
		
		The league `id` are **unique** in the API and competitions keep it across all `seasons`
		
		This endpoint contains a field named `coverage` that indicates for each season of a competition the data that are available."
    is_id: The id of the league
        season: The season of the league
        current: Values: `true` or `false`
        
    """
    url = f"https://api-american-football.p.rapidapi.com/leagues"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if season:
        querystring['season'] = season
    if current:
        querystring['current'] = current
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timezone(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of timezone set that can be used in the endpoint `games`.
		
		**Parameters:** This endpoint does not require any parameters."
    
    """
    url = f"https://api-american-football.p.rapidapi.com/timezone"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the list of all available seasons for all competitions.
		
		All seasons are only **4-digit keys**, so for a league whose season is `2018-2019` the season in the API will be `2018`.
		
		All `seasons` can be used in other endpoints as filters.
		
		**Parameters:** This endpoint does not require any parameters."
    
    """
    url = f"https://api-american-football.p.rapidapi.com/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-american-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

