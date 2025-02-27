import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def event_odds(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get odds by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/odds"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_events(timezone: int, indent_days: int, locale: str, sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of events by sport_id. <br> `STAGE_TYPE` - main event status, `STAGE` - more event status. [List of object statuses:](https://rapidapi.com/tipsters/api/flashlive-sports/tutorials/flashlive-list-of-object-statuses)"
    timezone: Time zone
        indent_days: Number of days from today. Ex: `-1` yesterday
        sport_id: Sport id, use `Sports list` endpoint
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/list"
    querystring = {'timezone': timezone, 'indent_days': indent_days, 'locale': locale, 'sport_id': sport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_main_odds(locale: str, sport_id: int, timezone: int, indent_days: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of main odds by sport and date"
    sport_id: Sport id, use `Sports list` endpoint
        timezone: Time zone
        indent_days: Number of days from today. Ex: `-1` yesterday
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/list-main-odds"
    querystring = {'locale': locale, 'sport_id': sport_id, 'timezone': timezone, 'indent_days': indent_days, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_report(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an event report by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/report"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_summary_results(event_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event summary results by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/summary-results"
    querystring = {'event_id': event_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_details_beta(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event details by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/details"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tables_tabs(locale: str, tournament_season_id: str, tournament_stage_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the available types (tabs) of standings"
    tournament_season_id: Season tournament id
        tournament_stage_id: Season tournament stage id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/tournaments/standings/tabs"
    querystring = {'locale': locale, 'tournament_season_id': tournament_season_id, 'tournament_stage_id': tournament_stage_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_tables(locale: str, tournament_stage_id: str, standing_type: str, tournament_season_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tournament tables"
    tournament_stage_id: Season tournament stage id
        tournament_season_id: Season tournament id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/tournaments/standings"
    querystring = {'locale': locale, 'tournament_stage_id': tournament_stage_id, 'standing_type': standing_type, 'tournament_season_id': tournament_season_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_stages_data(tournament_stage_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tournament data by tournament_stage_id"
    tournament_stage_id: Tournament stage id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/tournaments/stages/data"
    querystring = {'tournament_stage_id': tournament_stage_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_results_events(locale: str, tournament_stage_id: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team results by tournament_stage_id. Use the pagination"
    tournament_stage_id: Season tournament stage id
        page: Page number
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/tournaments/results"
    querystring = {'locale': locale, 'tournament_stage_id': tournament_stage_id, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stages_list(locale: str, sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of stages by sport ID"
    sport_id: Sport id, use `Sports list` endpoint
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/tournaments/stages"
    querystring = {'locale': locale, 'sport_id': sport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_popular_list(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of popular tournaments by sport ID"
    
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/tournaments/popular"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_list(locale: str, sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of tournaments by sport ID"
    sport_id: Sport id, use `Sports list` endpoint
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/tournaments/list"
    querystring = {'locale': locale, 'sport_id': sport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_data(locale: str, season_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get season data by season_id"
    season_id: Season tournament id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/tournaments/seasons/data"
    querystring = {'locale': locale, 'season_id': season_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_fixtures_events(tournament_stage_id: str, locale: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get upcoming events by tournament_stage_id, Use the pagination"
    tournament_stage_id: Season tournament stage id
        page: Page number
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/tournaments/fixtures"
    querystring = {'tournament_stage_id': tournament_stage_id, 'locale': locale, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def number_of_sport_events(locale: str, timezone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about the sport and the number of sport events"
    timezone: Time zone
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/sports/events-count"
    querystring = {'locale': locale, 'timezone': timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sports_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of sports"
    
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/sports/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_news(team_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of news by team ID"
    team_id: Team id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/teams/news"
    querystring = {'team_id': team_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_events(player_id: str, locale: str, sport_id: int, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the player events by player ID"
    player_id: Player id
        sport_id: Sport id, use `Sports list` endpoint
        page: Page number
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/players/alt-events"
    querystring = {'player_id': player_id, 'locale': locale, 'sport_id': sport_id, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_results_events(sport_id: int, locale: str, team_id: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team results by team ID. Use the pagination"
    sport_id: Sport id, use `Sports list` endpoint
        team_id: Team id
        page: Page number
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/teams/results"
    querystring = {'sport_id': sport_id, 'locale': locale, 'team_id': team_id, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_career(player_id: str, locale: str, sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player career by player ID"
    player_id: Player id
        sport_id: Sport id, use `Sports list` endpoint
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/players/career"
    querystring = {'player_id': player_id, 'locale': locale, 'sport_id': sport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_data(image_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a image player, team, tournament by IMAGE_ID"
    image_id: Image ID
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/images/data"
    querystring = {'image_id': image_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_transfers(team_id: str, locale: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team transfers by team ID"
    team_id: Team id
        page: Page number
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/teams/transfers"
    querystring = {'team_id': team_id, 'locale': locale, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_data(sport_id: int, player_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player data by player ID"
    sport_id: Sport id, use `Sports list` endpoint
        player_id: Player id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/players/data"
    querystring = {'sport_id': sport_id, 'player_id': player_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_data(locale: str, team_id: str, sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team data by team ID"
    team_id: Team id
        sport_id: Sport id, use `Sports list` endpoint
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/teams/data"
    querystring = {'locale': locale, 'team_id': team_id, 'sport_id': sport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_fixtures_events(team_id: str, locale: str, sport_id: int, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get upcoming events by team ID. Use the pagination"
    team_id: Team id
        sport_id: Sport id, use `Sports list` endpoint
        page: Page number
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/teams/fixtures"
    querystring = {'team_id': team_id, 'locale': locale, 'sport_id': sport_id, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ranking_list(sport_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of ranking by sport ID"
    sport_id: Sport id, use `Sports list` endpoint
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/rankings/list"
    querystring = {'sport_id': sport_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_player_events(sport_id: int, player_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest player events by player ID"
    sport_id: Sport id, use `Sports list` endpoint
        player_id: Player id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/players/last-events"
    querystring = {'sport_id': sport_id, 'player_id': player_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_squad(sport_id: int, locale: str, team_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get squad by team ID"
    sport_id: Sport id, use `Sports list` endpoint
        team_id: Team id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/teams/squad"
    querystring = {'sport_id': sport_id, 'locale': locale, 'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_transfers(player_id: str, locale: str, sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest player transfers by player ID"
    player_id: Player id
        sport_id: Sport id, use `Sports list` endpoint
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/players/transfers"
    querystring = {'player_id': player_id, 'locale': locale, 'sport_id': sport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ranking_data(locale: str, ranking_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a ranking data by ranking ID"
    ranking_id: Ranking id, use `Ranking list` endpoint
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/rankings/data"
    querystring = {'locale': locale, 'ranking_id': ranking_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_read_news(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the most read news"
    
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/news/most-read"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_news(locale: str, category_id: int, entity_id: str, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of news by category_id and entity_id"
    category_id: Category ID, use `News categories` endpoint
        entity_id: Entity ID, use `News categories` endpoint
        page: Page number
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/news/list"
    querystring = {'locale': locale, 'category_id': category_id, 'entity_id': entity_id, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_details(locale: str, article_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information about the news by article_id"
    article_id: Article ID
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/news/details"
    querystring = {'locale': locale, 'article_id': article_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def summary_news_data(article_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brief information about the news by article_id"
    article_id: Article ID
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/news/summary"
    querystring = {'article_id': article_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def multi_search(query: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search teams, tournaments and players. Results are shown as you type."
    query: Query, search request
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/search/multi-search"
    querystring = {'query': query, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def related_news(article_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get related news by article_id"
    article_id: Article ID
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/news/related"
    querystring = {'article_id': article_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_of_news(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get categories of news"
    
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/news/categories"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_top_news(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of top news"
    
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/news/top"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def changes_to_live_events(locale: str, sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get only new data into live-events by sport_id. Call every 5 seconds."
    sport_id: Sport id, use `Sports list` endpoint
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/live-update"
    querystring = {'locale': locale, 'sport_id': sport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_player_statistics_alt_basketball(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player statistics by event ID, (basketball)"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/player-statistics-alt"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_prematch_odds(sport_id: int, event_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event prematch odds by event ID"
    sport_id: Sport id, use `Sports list` endpoint
        event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/prematch-odds"
    querystring = {'sport_id': sport_id, 'event_id': event_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_data_no_duel(no_duel_event_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Data on the No-duel event, (golf)"
    no_duel_event_id: No duel event id (NO_DUEL_EVENT_ID)
        event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/no-duel-data"
    querystring = {'no_duel_event_id': no_duel_event_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_ball_ball_cricket(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a ball by ball on an event (cricket) by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/ball-by-ball"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_preview(event_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an event preview by event ID."
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/preview"
    querystring = {'event_id': event_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_last_change(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get new data keys. Call this point to see what data has been changed."
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/last-change"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_fall_wickets_cricket(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a fall of wickets on an event (cricket) by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/fall-of-wickets"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_scorecard_cricket(event_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a scorecard on an event (cricket) by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/scorecard"
    querystring = {'event_id': event_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_statistics(event_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event statistics by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/statistics"
    querystring = {'event_id': event_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_commentary(event_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a comment on an event by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/commentary"
    querystring = {'event_id': event_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def h2h_events(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get events between two teams by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/h2h"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_missing_players(event_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all missing players by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/missing-players"
    querystring = {'event_id': event_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_points_history(event_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the history of points by event ID. Point by Point"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/points-history"
    querystring = {'event_id': event_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_news(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/news"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def racing_details(locale: str, sport_id: int, timezone: int, tournament_template_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Racing details, (HORSE_RACING)"
    sport_id: Sport id, use `Sports list` endpoint
        timezone: Time zone
        tournament_template_id: Tournament template ID
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/racing-details"
    querystring = {'locale': locale, 'sport_id': sport_id, 'timezone': timezone, 'tournament_template_id': tournament_template_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_odds_live(book_id: int, locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live odds by event ID"
    book_id: Bookmaker id, 
        event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/live-odds"
    querystring = {'book_id': book_id, 'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_data_brief(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brief data by event ID. It is better to use Event Details and Event Data Brief endpoints"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/brief"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_starting_lineups(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get starting lineups by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/lineups"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_data(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all event data by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/data"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_statistics_alt(event_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event statistics by event ID (darts)"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/statistics-alt"
    querystring = {'event_id': event_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_throw_throw_darts(event_id: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get throw by throw (darts) by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/throw-by-throw"
    querystring = {'event_id': event_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_summary_incidents(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event summary incidents by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/summary-incidents"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_summary(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get summary of the event by event ID. Incidents of the match"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/summary"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_live_events(timezone: int, sport_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of live events. Use `/live-update` to get changes in live events!"
    timezone: Time zone
        sport_id: Sport id, use `Sports list` endpoint
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/live-list"
    querystring = {'timezone': timezone, 'sport_id': sport_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_commentary_alt_cricket(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a comment on an event (cricket) by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/commentary-alt"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_rounds_results_golf(locale: str, no_duel_event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rounds results (golf) by  and event ID, (golf)"
    no_duel_event_id: No duel event id (NO_DUEL_EVENT_ID)
        event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/rounds-results"
    querystring = {'locale': locale, 'no_duel_event_id': no_duel_event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_highlights_video(locale: str, event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get highlights video by event ID"
    event_id: Event id
        
    """
    url = f"https://flashlive-sports.p.rapidapi.com/v1/events/highlights"
    querystring = {'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flashlive-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

