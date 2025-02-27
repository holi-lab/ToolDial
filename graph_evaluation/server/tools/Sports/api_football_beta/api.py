import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fixtures_rounds(season: int, league: int, current: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the rounds for a league or a cup
		**Update Frequency** : Every day
		**Recommended Calls** : 1 call per day"
    season: The season of the league, "YYYY"
        league: The id of the league
        current: The current round only, "true", "false"
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/fixtures/rounds"
    querystring = {'season': season, 'league': league, }
    if current:
        querystring['current'] = current
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_fixtures_statistics(fixture: int, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the players statistics from one fixture
		**Update Frequency** : Every minute
		**Recommended Calls** : 1 call every minute for the fixtures in progress otherwise 1 call per day"
    fixture: The id of the fixture
        team: The id of the team
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/fixtures/players"
    querystring = {'fixture': fixture, }
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons_for_players(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available seasons for players statistics
		**Update Frequency** : Every day
		**Recommended Calls** : 1 call per day"
    
    """
    url = f"https://api-football-beta.p.rapidapi.com/players/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_statistics(team: int, season: int, league: int, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics for a team
		**Update Frequency** : Every 15 seconds.
		**Recommended Calls** : 1 call per minute for the teams who have at least one fixture in progress otherwise 1 call per week"
    team: The id of the team
        season: The season of the league, "YYYY"
        league: The id of the league
        date: The limit date, "YYYY-MM-DD"
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/teams/statistics"
    querystring = {'team': team, 'season': season, 'league': league, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_statistics(team: int, season: int, league: int, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics for a team
		**Update Frequency** : Every 15 seconds.
		**Recommended Calls** : 1 call per minute for the teams who have at least one fixture in progress otherwise 1 call per week"
    team: The id of the team
        season: The season of the league, "YYYY"
        league: The id of the league
        date: The limit date, "YYYY-MM-DD"
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/teams/statistics"
    querystring = {'team': team, 'season': season, 'league': league, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings(season: int, team: int=None, league: int=39, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the standings for a league or a team
		**Update Frequency** : Every hour
		**Recommended Calls** : 1 call per hour for the leagues or teams who have at least one fixture in progress otherwise 1 call per day"
    season: The season of the league, "YYYY"
        team: The id of the team
        league: The id of the league
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/standings"
    querystring = {'season': season, }
    if team:
        querystring['team'] = team
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams(name: str=None, search: str=None, is_id: int=None, league: int=39, season: int=2019, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available teams
		**Update Frequency** : Several times a week.
		**Recommended Calls** : 1 call per day"
    name: The name of the team
        search: The name or the country name of the team
        is_id: The id of the team
        league: The id of the league
        season: The season of the league
        country: The country name of the team
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/teams"
    querystring = {}
    if name:
        querystring['name'] = name
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    if league:
        querystring['league'] = league
    if season:
        querystring['season'] = season
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timezone(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available timezone in the API
		**Recommended Calls** : 1 call when you need"
    
    """
    url = f"https://api-football-beta.p.rapidapi.com/timezone"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available seasons
		**Update Frequency** :Each time a new league is added
		**Recommended Calls** : 1 call per day"
    
    """
    url = f"https://api-football-beta.p.rapidapi.com/leagues/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures(to: str=None, timezone: str=None, season: int=None, status: str=None, last: int=None, round: str=None, live: str=None, league: int=None, is_id: int=None, is_from: str=None, date: str='2020-02-06', next: int=None, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fixtures
		You can use several parameters at the same time (Check the documentation)
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    to: A valid date, "YYYY-MM-DD"
        timezone: Fails if field is not a result of the endpoint timezone
        season: The season of the league, "YYYY"
        status: The short status
        round: The round of the fixture
        live: "all" or by league id : "id-id-id..."
        league: The id of the league
        is_id: The id of the fixture
        is_from: A valid date, "YYYY-MM-DD"
        date: A valid date, "YYYY-MM-DD"
        team: The id of the team
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/fixtures"
    querystring = {}
    if to:
        querystring['to'] = to
    if timezone:
        querystring['timezone'] = timezone
    if season:
        querystring['season'] = season
    if status:
        querystring['status'] = status
    if last:
        querystring['last'] = last
    if round:
        querystring['round'] = round
    if live:
        querystring['live'] = live
    if league:
        querystring['league'] = league
    if is_id:
        querystring['id'] = is_id
    if is_from:
        querystring['from'] = is_from
    if date:
        querystring['date'] = date
    if next:
        querystring['next'] = next
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfers(team: int=None, player: int=1578, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available transfers for players and teams
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    team: The id of the team
        player: The id of the player
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/transfers"
    querystring = {}
    if team:
        querystring['team'] = team
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_h2h(h2h: str, league: int=None, status: str='ft', season: int=None, timezone: str=None, is_from: str=None, last: int=10, date: str=None, next: int=None, to: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get heads to heads between two teams
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the leagues, teams, fixtures who have at least one fixture in progress otherwise 1 call per day"
    h2h: The ids of the teams
        league: The id of the league
        status: The short status
        season: The season of the league, "YYYY"
        timezone: Fails if field is not a result of the endpoint timezone
        is_from: A valid date, "YYYY-MM-DD"
        date: A valid date, "YYYY-MM-DD"
        to: A valid date, "YYYY-MM-DD"
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/fixtures/headtohead"
    querystring = {'h2h': h2h, }
    if league:
        querystring['league'] = league
    if status:
        querystring['status'] = status
    if season:
        querystring['season'] = season
    if timezone:
        querystring['timezone'] = timezone
    if is_from:
        querystring['from'] = is_from
    if last:
        querystring['last'] = last
    if date:
        querystring['date'] = date
    if next:
        querystring['next'] = next
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_events(fixture: int, type: str=None, player: int=None, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get events from a fixture
		**Update Frequency** : Every 15 seconds
		**Recommended Calls** : 1 call per minute for the fixtures in progress otherwise 1 call per day"
    fixture: The id of the fixture
        type: The type of event, "goal, card..."
        player: The id of the player
        team: The id of the team
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/fixtures/events"
    querystring = {'fixture': fixture, }
    if type:
        querystring['type'] = type
    if player:
        querystring['player'] = player
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_statistics(fixture: int, team: int=None, type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the statistics for one fixture
		**Update Frequency** : Every minute
		**Recommended Calls** : 1 call every minute for the teams or fixtures who have at least one fixture in progress otherwise 1 call per day"
    fixture: The id of the fixture
        team: The id of the team
        type: The type of statistics, "Offsides..."
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/fixtures/statistics"
    querystring = {'fixture': fixture, }
    if team:
        querystring['team'] = team
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leagues(last: int=None, type: str=None, name: str=None, search: str=None, team: int=None, current: str=None, season: str=None, code: str=None, is_id: int=61, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available leagues and cups
		**Update Frequency** : Several times a day
		**Recommended Calls** : 1 call per hour"
    last: The X last leagues/cups added in the API
        type: The type of the league, "league", "cup"
        name: The name of the league
        search: The name or the country of the league
        team: The id of the team
        current: The state of the league
        season: The season of the league, "YYYY"
        code: The Alpha2 code of the country
        is_id: The id of the league
        country: The country name of the league
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/leagues"
    querystring = {}
    if last:
        querystring['last'] = last
    if type:
        querystring['type'] = type
    if name:
        querystring['name'] = name
    if search:
        querystring['search'] = search
    if team:
        querystring['team'] = team
    if current:
        querystring['current'] = current
    if season:
        querystring['season'] = season
    if code:
        querystring['code'] = code
    if is_id:
        querystring['id'] = is_id
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venues(name: str=None, is_id: int=556, country: str=None, city: str=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available venues
		**Update Frequency** : This endpoint is updated several times a week
		**Recommended Calls** : 1 call per day"
    name: The name of the venue
        is_id: The id of the venue
        country: The country name of the venue
        city: The city of the venue
        search: The name, city or the country of the venue
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/venues"
    querystring = {}
    if name:
        querystring['name'] = name
    if is_id:
        querystring['id'] = is_id
    if country:
        querystring['country'] = country
    if city:
        querystring['city'] = city
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries(search: str=None, name: str=None, code: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available countries
		**Update Frequency** : This endpoint is updated each time a new league from a country not covered by the API is added
		**Recommended Calls** : 1 call per day"
    search: The Alpha2 code of the country
        name: The name of the country
        code: The name of the country
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/countries"
    querystring = {}
    if search:
        querystring['search'] = search
    if name:
        querystring['name'] = name
    if code:
        querystring['code'] = code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coachs(search: str='tuchel', team: int=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the information about the coachs and their careers
		**Update Frequency** : Every day
		**Recommended Calls** : 1 call per day"
    search: The name of the coach
        team: The id of the team
        is_id: The id of the coach
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/coachs"
    querystring = {}
    if search:
        querystring['search'] = search
    if team:
        querystring['team'] = team
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_season_statistics(page: int=None, season: int=None, team: int=None, league: int=None, is_id: int=276, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all players statistics from teams, leagues and seasons
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    page: Use for the pagination (Default : 1)
        season: The season of the league, "YYYY"
        team: The id of the team
        league: The id of the league
        is_id: The id of the player
        search: The name of the player
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/players"
    querystring = {}
    if page:
        querystring['page'] = page
    if season:
        querystring['season'] = season
    if team:
        querystring['team'] = team
    if league:
        querystring['league'] = league
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bookmakers(is_id: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bookmakers
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    is_id: The id of the bookmaker
        search: The name of the bookmaker
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/odds/bookmakers"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_players_statistics(fixture: int, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the players statistics from one fixture
		**Update Frequency** : Every minute
		**Recommended Calls** : 1 call every minute for the fixtures in progress otherwise 1 call per day"
    fixture: The id of the fixture
        team: The id of the team
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/fixtures/players"
    querystring = {'fixture': fixture, }
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sidelined(coach: int=None, player: int=276, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available sidelined for a player or a coach
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    coach: The id of the coach
        player: The id of the player
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/sidelined"
    querystring = {}
    if coach:
        querystring['coach'] = coach
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sidelined(coach: int=None, player: int=276, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available sidelined for a player or a coach
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    coach: The id of the coach
        player: The id of the player
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/sidelined"
    querystring = {}
    if coach:
        querystring['coach'] = coach
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds(date: str='2020-05-15', page: int=None, timezone: str=None, bookmaker: int=None, fixture: int=None, season: int=None, bet: int=None, league: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get odds from fixtures or leagues
		**Update Frequency** : Every day
		**Recommended Calls** : 1 call per day"
    date: A valid date (YYYY-MM-DD)
        page: Use for the pagination (Default : 1)
        timezone: A valid timezone from the endpoint \"Timezone\"
        bookmaker: The id of the bookmaker
        fixture: The id of the fixture
        season: the season of the league, "YYYY"
        bet: The id of the bet
        league: The id of the league
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/odds"
    querystring = {}
    if date:
        querystring['date'] = date
    if page:
        querystring['page'] = page
    if timezone:
        querystring['timezone'] = timezone
    if bookmaker:
        querystring['bookmaker'] = bookmaker
    if fixture:
        querystring['fixture'] = fixture
    if season:
        querystring['season'] = season
    if bet:
        querystring['bet'] = bet
    if league:
        querystring['league'] = league
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bets(search: str=None, is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available bets.
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    search: The name of the bet
        is_id: The id of the bet
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/odds/bets"
    querystring = {}
    if search:
        querystring['search'] = search
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_scorers(season: int, league: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get the 20 best players for a league or cup
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    season: The season of the league, "YYYY"
        league: The id of the league
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/players/topscorers"
    querystring = {'season': season, 'league': league, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_scorers(season: int, league: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get the 20 best players for a league or cup
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    season: The season of the league, "YYYY"
        league: The id of the league
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/players/topscorers"
    querystring = {'season': season, 'league': league, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mapping(page: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available odds
		**Update Frequency** : Every day
		**Recommended Calls** : 1 call per day"
    page: Use for the pagination
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/odds/mapping"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_predictions(fixture: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get predictions about a fixture
		**Update Frequency** : Every hour
		**Recommended Calls** : 1 call per hour for the fixtures in progress otherwise 1 call per day"
    
    """
    url = f"https://api-football-beta.p.rapidapi.com/predictions"
    querystring = {'fixture': fixture, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_lineups(fixture: int, type: str=None, player: int=None, team: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the lineups for one fixture
		**Update Frequency** : Every 15 minutes.
		**Recommended Calls** : 1 call every 15 minutes for the fixtures in progress otherwise 1 call per day"
    fixture: The id of the fixture
        type: The type, "startxi..."
        player: The id of the player
        team: The id of the team
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/fixtures/lineups"
    querystring = {'fixture': fixture, }
    if type:
        querystring['type'] = type
    if player:
        querystring['player'] = player
    if team:
        querystring['team'] = team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trophies(coach: int=None, player: int=276, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available trophies for a player or a coach
		**Update Frequency** : Several times a week
		**Recommended Calls** : 1 call per day"
    coach: The id of the coach
        player: The id of the player
        
    """
    url = f"https://api-football-beta.p.rapidapi.com/trophies"
    querystring = {}
    if coach:
        querystring['coach'] = coach
    if player:
        querystring['player'] = player
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "api-football-beta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

