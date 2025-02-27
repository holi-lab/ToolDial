import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def standings_corrections(is_id: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "n case a team got points corrections they can be found here. It will return an array of correction objects on a per stage basis. In some occasions a stage proceeds with standings from an earlier stage. In these situation we will also provide correction objects so you can easily understand how a standing for a stage has been calculated."
    id: The ID of the Season.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/corrections/season/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_leagues(page: str='1', include: str=None, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you can get an overview of all the leagues. | Popular includes: country,season"
    page: The number of the page you want to browse. 
        include: Optional includes to enrich the API response.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/leagues"
    querystring = {}
    if page:
        querystring['page'] = page
    if include:
        querystring['include'] = include
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aggregated_assistscorers_by_season_id(include: str, is_id: int, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Topscorers endpoint returns the Aggregated Assistscorers by Season. This means that all stages are summed, also preliminary stages."
    include: Required includes to enrich the API response.
        id: The id of the season you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/seasons/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aggregated_cardscorers_by_season_id(include: str, is_id: int, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Topscorers endpoint returns the Aggregated Cardscores by Season. This means that all stages are summed, also preliminary stages."
    include: Required includes to enrich the API response.
        id: The id of the season you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/seasons/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aggregated_goalscorers_by_season_id(is_id: int, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Aggregated Goalscores endpoint returns the Goalscores by Stage of the Season."
    id: The id of the season you want data for.
        include: Required includes to enrich the API response.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/seasons/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_aggregated_topscorers_by_season_id(is_id: int, tz: str='Europe/Amsterdam', include: str='aggregatedGoalscorers.player,aggregatedGoalscorers.team,aggregatedCardscorers.player,aggregatedCardscorers.team,aggregatedAssistscorers.player,aggregatedAssistscorers.team', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Topscorers endpoint returns the Aggregated Topscorers by Season. This means that all stages are summed, also preliminary stages."
    id: The id of the season you want data for.
        tz: Override the default timezone with your own timezone.
        include: Optional includes to enrich the API response
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/topscorers/season/{is_id}/aggregated"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_bookmakers(tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives you all the available Bookmakers."
    tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/bookmakers"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def round_by_id(is_id: str, tz: str='Europe/Amsterdam', include: str='fixtures,season,league', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Leagues can be split up in Rounds representing a week a game is played in. With this endpoint we give you the ability to request data for a single Round  |  Popular includes: fixtures.visitorTeam,fixtures.localTeam"
    id: The ID of the Round
        tz: Override the default timezone with your own timezone. 
        include: Optional includes to enrich the API response.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/rounds/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_continents(tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Continents endpoint helps you with assigning Countries and Leagues to the part of the world (Continent) they belong to."
    tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/continents"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_by_date(date: str, leagues: str='8', include: str='localTeam,visitorTeam,league,season', bookmakers: str='2', markets: str='1', tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Fixture endpoint provides information about Games in particular Leagues. With this endpoint you can get the fixtures by Date. | Popular includes: localTeam,visitorTeam,events"
    date: The desired date
        leagues: Filter based on league ids. Example: 8,9
        include: Optional includes to enrich the API response.
        bookmakers: Filter odds based on a comma separated list of bookmaker ids. Example: 2,15
        markets: Filter odds based on a comma separated list of market ids. Example: 1,10
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/fixtures/date/{date}"
    querystring = {}
    if leagues:
        querystring['leagues'] = leagues
    if include:
        querystring['include'] = include
    if bookmakers:
        querystring['bookmakers'] = bookmakers
    if markets:
        querystring['markets'] = markets
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_odds_markets(tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives you all the available Markets."
    tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/markets"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_by_id_list(idlist: str, bookmakers: str='2', tz: str='Europe/Amsterdam', include: str='localTeam,visitorTeam,events', markets: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Fixture endpoint provides information about Games in particular Leagues. With this endpoint you can get multiple fixtures by IDLIST. | Popular includes: localTeam,visitorTeam"
    idlist: List of ID's of fixtures
        bookmakers: Filter odds based on a comma separated list of bookmaker ids. Example: 2,15
        tz: Override the default timezone with your own timezone.
        include: Optional includes to enrich the API response.
        markets: Filter odds based on a comma separated list of market ids. Example: 1,10
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/fixtures/multi/{idlist}"
    querystring = {}
    if bookmakers:
        querystring['bookmakers'] = bookmakers
    if tz:
        querystring['tz'] = tz
    if include:
        querystring['include'] = include
    if markets:
        querystring['markets'] = markets
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rounds_by_season_id(is_id: str, include: str='league', tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Leagues can be split up in Rounds representing a week a game is played in. With this endpoint we give you the ability to request data for a single Round as well as all Rounds of a Season. | Popular includes: fixtures.visitorTeam,fixtures.localTeam"
    id: The ID of the Season
        include: Optional includes to enrich the API response.
        tz: Override the default timezone with your own timezone. Example: Europe/Amsterdam
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/rounds/season/{is_id}"
    querystring = {}
    if include:
        querystring['include'] = include
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_seasons(page: str='1', include: str=None, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you can get an overview of all the seasons. | Popular includes: league,stages,rounds"
    page: The number of the page you want to browse. 
        include: Optional includes to enrich the API response.
        tz: Override the default timezone with your own timezone. 
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/seasons"
    querystring = {}
    if page:
        querystring['page'] = page
    if include:
        querystring['include'] = include
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_currently_inplay(tz: str='Europe/Amsterdam', bookmakers: str='2', leagues: str=None, markets: str='1', include: str='localTeam,visitorTeam,events', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Fixture endpoint provides information about Games in particular Leagues. With this endpoint you can get all the Livescores that are in-play. Popular includes | localTeam,visitorTeam,events"
    tz: Override the default timezone with your own timezone.
        bookmakers: Filter odds based on a comma separated list of bookmaker ids. Example: 2,15
        leagues: Filter based on league ids. Example: 8,9
        markets: Filter odds based on a comma separated list of market ids. Example: 1,10
        include: Optional includes to enrich the API response.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/livescores/now"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if bookmakers:
        querystring['bookmakers'] = bookmakers
    if leagues:
        querystring['leagues'] = leagues
    if markets:
        querystring['markets'] = markets
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_inplay_by_league(leagues: str, bookmakers: str='2', timezone: str='Europe/Amsterdam', market: str='1', include: str='localTeam,visitorTeam,events', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Fixture endpoint provides information about Games in particular Leagues. With this endpoint you can get all the Livescores that are in-play per League ID. Popular includes | localTeam,visitorTeam,events"
    leagues: Filter based on league ids. Example: 8,9
        bookmakers: Filter odds based on a comma separated list of bookmaker ids. Example: 2,15
        timezone: Override the default timezone with your own timezone. 
        market: Filter odds based on a comma separated list of market ids. Example: 1,10
        include: Optional includes to enrich the API response.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/livescores/now"
    querystring = {'leagues': leagues, }
    if bookmakers:
        querystring['bookmakers'] = bookmakers
    if timezone:
        querystring['timezone'] = timezone
    if market:
        querystring['market'] = market
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def assistscorers_by_season_id(is_id: int, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Assistscorers endpoint returns the Assistscorers by Stage of the Season."
    id: The id of the season you want data for.
        include: Required includes to enrich the API response.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/seasons/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_of_today(tz: str='Europe/Amsterdam', markets: str=None, bookmakers: str=None, leagues: str=None, include: str='localTeam,visitorTeam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Fixture endpoint provides information about Games in particular Leagues. With this endpoint you can get all the Livescores of the current day. | Popular includes: localTeam,visitorTeam"
    tz: Override the default timezone with your own timezone.
        markets: Filter odds based on a comma separated list of market ids. Example: 1,10
        bookmakers: Filter odds based on a comma separated list of bookmaker ids. Example: 2,15
        leagues: Filter based on league ids. Example: 8,9
        include: Optional includes to enrich the API response.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/livescores"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if markets:
        querystring['markets'] = markets
    if bookmakers:
        querystring['bookmakers'] = bookmakers
    if leagues:
        querystring['leagues'] = leagues
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def benched_players_by_fixture_id(is_id: str, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives you the benched Players by fixture ID"
    id: The id of the fixtures you want data for.
        include: Required includes to enrich the API response.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/fixtures/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bookmaker_by_id(is_id: int, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives information about the Bookmaker by ID."
    id: The id of the bookmaker you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/bookmakers/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_line_ups_by_player_id(is_id: str, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives you the line-up information of one Player by ID."
    id: The id of the player you want data for.
        include: Required includes to enrich the API response.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/players/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_by_id(is_id: str, page: str='1', tz: str='Europe/Amsterdam', include: str='league,stages,rounds', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Seasons endpoint provides you details about for example  Name, League ID, Year and if the Season is Active Yes or No. One of the main features of the Seasons endpoint is the ability to retrieve full Season Games, (history) Fixtures, Results or Upcoming Fixtures.  Popular includes: stages,rounds,results"
    id: The ID of the Season
        page: The number of the page you want to browse. Can be used by adding &page=2 etc to your request url.
        tz: Override the default timezone with your own timezone.
        include: Optional includes to enrich the API response.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/seasons/{is_id}"
    querystring = {}
    if page:
        querystring['page'] = page
    if tz:
        querystring['tz'] = tz
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goalcorers_by_season_id(is_id: int, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Goalscorers endpoint returns the Goalscorers by Stage of the Season."
    id: The id of the season you want data for.
        include: Required includes to enrich the API response.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/seasons/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cardscorers_by_season_id(is_id: str, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Topscorers endpoint returns the Cardscorers by Stage of the Season."
    id: The ID of the season
        include: Requried includes to enrich the API response.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/seasons/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coach_by_id(is_id: int, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you can get detailed information of one Coach by ID."
    id: The id of the coach you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/coaches/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def commentaries_by_fixture_id(is_id: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives you the Commentaries by Fixture ID."
    id: The ID of the fixture you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/commentaries/fixture/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def continent_by_id(is_id: int, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives you information of one Continent by ID."
    id: The ID of the continent you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/continents/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_by_id(is_id: int, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives you information of one Country by ID."
    id: The id of the country you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/countries/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_by_fixture_id(include: str, is_id: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives you the events by Fixture ID."
    include: Requried includes to enrich the API response.
        id: The id of the fixture you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/fixtures/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixture_by_id(is_id: str, tz: str='Europe/Amsterdam', markets: str='1', include: str='localTeam,visitorTeam,events,lineup, localCoach,visitorCoach', bookmakers: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you can access the fixture by ID."
    id: ID of fixture
        tz: Override the default timezone with your own timezone. 
        markets: Filter odds based on a comma separated list of market ids.
        include: Optional includes to enrich the API response.
        bookmakers: Filter odds based on a comma separated list of bookmaker ids.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/fixtures/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if markets:
        querystring['markets'] = markets
    if include:
        querystring['include'] = include
    if bookmakers:
        querystring['bookmakers'] = bookmakers
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def head_2_head_by_team_id_s(teamb_id: str, teama_id: str, tz: str='Europe/Amsterdam', include: str='localTeam,visitorTeam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides you with all previous Games between 2 Teams. The Head 2 Head endpoint can give you a complete overview of the historical Games between 2 teams. Popular includes:  events, line-ups"
    teamb_id: The id of the team you want data for.
        teama_id: The id of the team you want data for.
        tz: Override the default timezone with your own timezone.
        include: Optional includes to enrich the API response.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/head2head/{teama_id}/{teamb_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_schedule(is_id: str, tz: str='Europe/Amsterdam', page: str='1', include: str='fixtures,rounds', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Seasons endpoint provides you with details about for example the Name, League ID, Year and if the Season is Active Yes or No.  One of the main features of the Seasons endpoint is the ability to retrieve full Season Games, (history) Fixtures, Results or Upcoming Fixtures. | Popular includes: fixtures.localTeam,fixtures.visitorTeam,rounds"
    id: The ID of the Season
        tz: Override the default timezone with your own timezone.
        page: The number of the page you want to browse. Can be used by adding &page=2 etc to your request url.
        include: Optional includes to enrich the API response.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/seasons/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if page:
        querystring['page'] = page
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_between_2_dates(startdate: str, enddate: str, bookmakers: str='2', leagues: str='8', markets: str='1', tz: str='Europe/Amsterdam', include: str='localTeam,visitorTeam,season', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you can get the fixtures between two dates: StartDate and EndDate."
    startdate: The first date of the date range
        enddate: The second date of the date range
        bookmakers: Filter odds based on a comma separated list of bookmaker ids.
        leagues: Filter based on league ids. 
        markets: Filter odds based on a comma separated list of market ids.
        tz:  Override the default timezone with your own timezone. 
        include: Optional includes to enrich the API response.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/fixtures/between/{startdate}/{enddate}"
    querystring = {}
    if bookmakers:
        querystring['bookmakers'] = bookmakers
    if leagues:
        querystring['leagues'] = leagues
    if markets:
        querystring['markets'] = markets
    if tz:
        querystring['tz'] = tz
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def inplay_odds_by_fixture_id(is_id: int, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Odds are used to add betting functionality to your application. This endpoint gives you the Live odds by Fixture ID."
    id: The id of the fixture you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/odds/inplay/fixture/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def last_updated_fixtures(markets: str='1', include: str='localTeam,visitorTeam,events', leagues: str='8', tz: str='Europe/Amsterdam', bookmakers: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Fixture endpoint provides information about Games in particular Leagues. With this endpoint you can get information about the fixtures that are last updated. | Popular includes: localTeam,visitorTeam,events"
    markets: Filter odds based on a comma separated list of market ids. Example: 1,10
        include: Optional includes to enrich the API response.
        leagues: Filter based on league ids. Example: 8,9
        tz: Override the default timezone with your own timezone. 
        bookmakers: Filter odds based on a comma separated list of bookmaker ids. Example: 2,15
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/fixtures/updates"
    querystring = {}
    if markets:
        querystring['markets'] = markets
    if include:
        querystring['include'] = include
    if leagues:
        querystring['leagues'] = leagues
    if tz:
        querystring['tz'] = tz
    if bookmakers:
        querystring['bookmakers'] = bookmakers
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_by_id(is_id: str, tz: str='Europe/Amsterdam', include: str='country,season,seasons', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Leagues endpoint is for most customers the starting point. With this endpoint you can get an overview of one league by ID."
    id: The id of the league you want data for.
        tz: The timezone settings
        include: Optional includes to enrich the API response.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/leagues/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_statistics_by_season_id(is_id: int, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns you all Season Statistics by Season. | Popular includes: topscorer,mostgoalteam,mostcornerteam"
    id: The ID of the Season
        include: Required include to get Statistics
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/seasons/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def line_up_by_fixture_id(is_id: str, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives you the line-up information of a Fixture ID."
    id: The id of the fixture you want data for.
        include: Required includes to enrich the API response.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/fixtures/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sidelined_by_player_id(is_id: str, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the Sidelined fixtures of a single Player. Sidelined means the fixtures where the Player was injured or suspended."
    id: The ID of the Player 
        include: Required include to get Sidelined by Player ID
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/players/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_statistics_by_fixture_id(is_id: int, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you can access all the statistics per team, player and season on various levels. This endpoint gives you the match statistics of a Fixture ID."
    id: The id of the fixture you want data for.
        include: Required includes to enrich the API response.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/fixtures/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds_market_by_id(is_id: int, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is very useful for administrative purposes to check and structure the Bookmakers and Markets availability. This endpoint gives information about the Markets by ID."
    id: The id of the market you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/markets/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_by_id(is_id: int, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides you with detailed Player and Coach information. With this endpoint you will be able to build a complete Player and Coach Profile. With this endpoint you can get detailed information of one Player by ID. | Popular includes: trophies,position,team,country"
    id: The id of the player you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/players/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_statistics_by_fixture_id(is_id: str, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you can access all the statistics per team, player and season on various levels. This endpoint gives you all the Player statistics of a Fixture ID."
    id: The id of the fixture you want data for.
        include: Required includes to enrich the API response.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/fixtures/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_statistics_by_player_id(is_id: int, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you can access all the statistics per team, player and season on various levels. This endpoint gives you all the statistics by Player ID."
    id: The id of the player you want data for.
        include: Required includes to enrich the API response.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/players/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_statistics_by_player_id_and_season_id(include: str, is_id: int, seasons: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With This Endpoint You Can Access All The Statistics Per Team, Player And Season On Various Levels. This Endpoint Gives You All The Statistics By Player ID and Season ID."
    include: Required includes to enrich the API response.
        id: The id of the player you want data for.
        seasons: The id of the seasons you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/players/{is_id}"
    querystring = {'include': include, 'seasons': seasons, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stage_by_id(is_id: str, include: str='fixtures', tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Leagues and Seasons all over the world can have a different set up. The Stages endpoint can help you to define the current Stage or set up of multiple Stages of a particular League/Season. | Popular includes:  season,league,fixtures"
    id: The ID of the Stage.
        include: Optional includes to enrich the API response.
        tz: Override the default timezone with your own timezone. 
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/stages/{is_id}"
    querystring = {}
    if include:
        querystring['include'] = include
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stages_by_season_id(is_id: str, tz: str='Europe/Amsterdam', include: str='league', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Leagues And Seasons All Over The World Can Have A Different Set Up. The Stages Endpoint Can Help You To Define The Current Stage Or Set Up Of Multiple Stages Of A Particular League/Season. | Popular Includes: season,league,fixtures"
    id: The ID of the Season
        tz: Override the default timezone with your own timezone.
        include: Optional includes to enrich the API response.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/stages/season/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pre_match_odds_by_fixture_id(is_id: int, include: str, markets: str='1', tz: str='Europe/Amsterdam', bookmakers: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Odds are used to add betting functionality to your application. This endpoint gives you the Pre-match odds by Fixture ID."
    id: The id of the fixture you want data for.
        include: Required includes to enrich the API response.
        markets: Filter odds based on a comma separated list of market ids. Example: 1,10
        tz: Override the default timezone with your own timezone.
        bookmakers: Filter odds based on a comma separated list of bookmaker ids. Example: 2,15
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/fixtures/{is_id}"
    querystring = {'include': include, }
    if markets:
        querystring['markets'] = markets
    if tz:
        querystring['tz'] = tz
    if bookmakers:
        querystring['bookmakers'] = bookmakers
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings_by_season_id(is_id: str, include: str='standings.team', stage_ids: str=None, group_ids: str=None, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Standings represent the rankings of Teams in the different Leagues they participate. Responses of the Standings endpoint can be returned in 2 formats, depending on the League setup. For ‘normal’ Leagues the response format is different compared to Cups. | Popular includes: standings.team,standings.stage"
    id: The ID of the season
        include: Optional includes to enrich the API response.
        stage_ids: Optional comma separated list of Stage ID's
        group_ids: Optional comma separated list of Group ID's
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/standings/season/{is_id}"
    querystring = {}
    if include:
        querystring['include'] = include
    if stage_ids:
        querystring['stage_ids'] = stage_ids
    if group_ids:
        querystring['group_ids'] = group_ids
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings_by_season_id_and_round_id(seasonid: str, roundid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Standings represent the rankings of Teams in the different Leagues they participate. Standings by Season ID and Round ID will return the standings for a given round. It will calculate the games played up until the given Round ID and provide a standings result for it."
    seasonid: The ID of the Season
        roundid: The ID of the Round
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/standings/season/{seasonid}/round/{roundid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_by_season_id(is_id: str, include: str=None, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With the Teams endpoint you can find all Team Details you need. You can think if information about when the Team is founded, Logo, Team Name, Short Name. With this endpoint you can get an overview of all the Teams by Season ID."
    id: The ID of the Season
        include: Optional includes to enrich the API response.
        tz: Override the default timezone with your own timezone
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/teams/season/{is_id}"
    querystring = {}
    if include:
        querystring['include'] = include
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_by_id(is_id: int, tz: str='Europe/Amsterdam', include: str='country,coach,venue', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With the Teams endpoint you can find all Team Details you need. You can think if information about when the Team is founded, Logo, Team Name, Short Name etc. | Popular includes: league,upcoming"
    id: The ID of the Team
        tz: Override the default timezone with your own timezone. 
        include: Optional includes to enrich the API response.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/teams/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def video_highlights_by_fixture_id(is_id: int, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives you the Video Highlights by fixture {ID}. Video Highlights is a community feature, which basically means we have no control over the availability of videos. So, we cannot guarantee the video you are looking for is available."
    id: The id of the fixtures you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/highlights/fixture/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venues_by_season_id(is_id: int, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides you with detailed Venue information like Name, City, Capacity, Address and even a Venue image. With this endpoint you can get detailed information of all the Venues by Season  ID."
    id: The id of the season you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/venues/season/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venues_by_id(is_id: int, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides you with detailed Venue information like Name, City, Capacity, Address and even a Venue image. With this endpoint you can get detailed information of one Venue by ID."
    id: The id of the venue you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/venues/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tv_stations_by_fixture_id(is_id: int, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The TV Stations endpoint is about TV Stations broadcasting the Football Games. Per Fixture, the endpoints returns all TV Stations that broadcast the Game."
    id: The id of the fixtures you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/tvstations/fixture/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trends_by_fixture_id(is_id: int, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you can get the Trend Analyses of a single Fixture. Please note it's also possible to use trends as include on Fixture level."
    id: The id of the fixture you want data for.
        include: The trends include to get it of a single Fixture
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/fixtures/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_squad_by_season_id(teamid: str, seasonid: str, tz: str='Europe/Amsterdam', include: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you can get Team Squad of one single Team by ID and  Season by ID. |  Popular includes: player"
    teamid: The ID of the Team
        seasonid: The ID of the Season
        tz: Override the default timezone with your own timezone. 
        include: Optional includes to enrich the API response.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/squad/season/{seasonid}/team/{teamid}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfers_by_player_id(include: str, is_id: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you can can get all the Transfers of a single Player by ID."
    include: Required includes to enrich the API response.
        id: The id of the player you want data for.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/players/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_statistics_by_team_id_and_season_id(is_id: str, season: str, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the Team Performance Statistics by Team ID and Season ID"
    id: The ID of the Team
        season: The ID of the Season
        include: Required include to show the stats
        tz: Override the default timezone with your own timezone. 
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/teams/{is_id}"
    querystring = {'season': season, 'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_season_statistics_by_team_id(is_id: int, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return all Season Statistics of All available Leagues/Seasons of the Team ID. So also for historical Seasons"
    id: The ID of the Team
        include: Required include to get the stats of the Team
        tz: Override the default timezone with your own timezone. 
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/teams/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_between_2_dates_for_one_team(enddate: str, teamid: int, startdate: str, include: str='localTeam,visitorTeam,season', leagues: str='8', markets: str='1', bookmakers: str='2', tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you can get the fixtures between two dates for one team."
    enddate: The second date of the date range
        startdate: The first date of the date range
        include: Optional includes to enrich the API response.
        leagues: Filter based on league ids. 
        markets: Filter odds based on a comma separated list of market ids.
        bookmakers: Filter odds based on a comma separated list of bookmaker ids.
        tz:  Override the default timezone with your own timezone. 
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/fixtures/between/{startdate}/{enddate}/{teamid}"
    querystring = {}
    if include:
        querystring['include'] = include
    if leagues:
        querystring['leagues'] = leagues
    if markets:
        querystring['markets'] = markets
    if bookmakers:
        querystring['bookmakers'] = bookmakers
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_standings_by_season_id(is_id: str, group_id: str=None, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Live Standings represent the rankings of Teams in the different Leagues they participate in real-time. Responses of the Standings endpoint can be returned in 2 formats, depending on the League setup. For ‘normal’ Leagues the response format is different compared to Cups."
    id: The ID of the Season.
        group_id: Optional to add the group ID when  available. group_id=185 for season 892 you will only get Group A standings for WorldCup 2018 season.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/standings/season/live/{is_id}"
    querystring = {}
    if group_id:
        querystring['group_id'] = group_id
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_topscorers_by_season_id(is_id: int, tz: str='Europe/Amsterdam', include: str='goalscorers.player,goalscorers.team,cardscorers.player,cardscorers.team,assistscorers.player,assistscorers.team', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Topscorers endpoint returns the Topscorers by Stage of the Season."
    id: The id of the season you want data for.
        tz: Override the default timezone with your own timezone. 
        include: Optional includes to enrich the API response.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/topscorers/season/{is_id}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_league_by_name(search: str, tz: str='Europe/Amsterdam', include: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns an array of Leagues that meet your search criteria including the active season ID. Use full league name for better results."
    search: The name of league you want to search
        tz:  Override the default timezone with your own timezone.
        include: Optional includes to enrich the API response. Optional includes are: country,season,seasons.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/leagues/search/{search}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_team_by_name(search: str, tz: str='Europe/Amsterdam', include: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns an array of teams that meet your search criteria. Use full team name for better results"
    search: The name of the team you want to search
        tz: Optional includes to enrich the API response. Optional includes are: counrty,stats and many more. Please check our docs for more information.
        include: Optional includes to enrich the API response. Optional includes are: counrty,stats and many more. Please check our docs for more information.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/teams/search/{search}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_player_by_name(search: str, tz: str='Europe/Amsterdam', include: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns an array of players that meet your search criteria. Use full player name for better results. For example 'Lionel%20Messi'"
    search: The name of the player you want to search
        tz: Override the default timezone with your own timezone.
        include: Optional includes to enrich the API response. Optional includes are: position,team,stats,trophies,sidelined,transfers,lineups,country
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/players/search/{search}"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if include:
        querystring['include'] = include
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def covid_19_information_per_league(include: str, page: str='1', tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can include information about COVID-19 on our Countries and Leagues endpoint. This includes the following information:
		
		1. You will get information like which leagues in the countries are affected.
		2. The expected start date per league.
		3. The official league status: None, Playing or Cancelled.
		4. Additional COVID-19 information."
    include: Optional includes to enrich the API response.
        page: The number of the page you want to browse. 
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/leagues"
    querystring = {'include': include, }
    if page:
        querystring['page'] = page
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def covid_19_information_per_country(include: str, tz: str='Europe/Amsterdam', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can include information about COVID-19 on our Countries and Leagues endpoint. This includes the following information:
		
		1. You will get information like which leagues in the countries are affected.
		2. The expected start date per league.
		3. The official league status: None, Playing or Cancelled.
		4. Additional COVID-19 information."
    include: Optional includes to enrich the API response.
        tz: Override the default timezone with your own timezone.
        page: The number of the page you want to browse. 
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/countries"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def active_season_by_team_id(is_id: int, include: str, tz: str='Europe/Amsterdam', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint you can get an overview all the active seasons of one Team by ID"
    id: The id of the team you want data for.
        include: Required includes to enrich the API response.
        tz: Override the default timezone with your own timezone.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/teams/{is_id}"
    querystring = {'include': include, }
    if tz:
        querystring['tz'] = tz
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_countries(tz: str='Europe/Amsterdam', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Countries endpoint provides you Country information like for example its Flag, IsoCode, Continent and other related Country information."
    tz: Override the default timezone with your own timezone.
        page: The number of the page you want to browse. Can be used by adding &page=2 etc to your request url.  You can also change the number of results per page to max. 150 by adding &per_page=150 to your request.
        
    """
    url = f"https://football-pro.p.rapidapi.com/api/v2.0/countries"
    querystring = {}
    if tz:
        querystring['tz'] = tz
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

