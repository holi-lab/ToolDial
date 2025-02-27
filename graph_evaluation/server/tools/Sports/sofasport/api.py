import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def number_live_events(sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get number of live events in the category by sport ID"
    sport_id: Sport id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/categories/number-live"
    querystring = {'sport_id': sport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channels_data(channel_id: int, event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get channels data"
    channel_id: Channel id
        event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/tv/channel-data"
    querystring = {'channel_id': channel_id, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_info(tournament_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tournament info by tournament id"
    tournament_id: Tournament id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/tournaments/info"
    querystring = {'tournament_id': tournament_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_media(unique_tournament_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media of the unique tournament"
    unique_tournament_id: Unique tournament id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/unique-tournaments/media"
    querystring = {'unique_tournament_id': unique_tournament_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schedule_live(sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live events schedule"
    sport_id: Sport id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/schedule/live"
    querystring = {'sport_id': sport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_channels_stage(stage_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tv country channels (stage) by stage_id"
    stage_id: Stage id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/tv/stage-country-channels"
    querystring = {'stage_id': stage_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def races_motorsport(page: int, course_events: str, team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get races by team ID (motorsport)"
    page: page
        team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/races"
    querystring = {'page': page, 'course_events': course_events, 'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unique_tournaments_list(category_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of unique tournaments by category id"
    category_id: Category id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/unique-tournaments"
    querystring = {'category_id': category_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_data(tournament_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tournament data by tournament id"
    tournament_id: Tournament id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/tournaments/data"
    querystring = {'tournament_id': tournament_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unique_tournament_data(unique_tournament_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get unique tournament data by unique tournament id"
    unique_tournament_id: Unique tournament id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/unique-tournaments/data"
    querystring = {'unique_tournament_id': unique_tournament_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_rankings(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team rankings by team ID"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/rankings"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unique_tournament_logo(unique_tournament_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get unique tournament logo by unique tournament id"
    unique_tournament_id: Unique tournament id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/unique-tournaments/logo"
    querystring = {'unique_tournament_id': unique_tournament_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unique_tournament_seasons(unique_tournament_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get seasons of the unique tournament"
    unique_tournament_id: Unique tournament id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/unique-tournaments/seasons"
    querystring = {'unique_tournament_id': unique_tournament_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ranking_by_season(year: int, ranking: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current by season"
    year: year
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/rankings/season"
    querystring = {'year': year, 'ranking': ranking, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ranking_current(ranking: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current rating"
    
    """
    url = f"https://sofasport.p.rapidapi.com/v1/rankings/current"
    querystring = {'ranking': ranking, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournaments_list(category_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of tournaments by category id"
    category_id: Category id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/tournaments"
    querystring = {'category_id': category_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_events(page: int, tournament_id: int, course_events: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get events of the tournament"
    page: page
        tournament_id: Unique tournament id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/tournaments/events"
    querystring = {'page': page, 'tournament_id': tournament_id, 'course_events': course_events, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def featured_events(unique_tournament_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get featured events of the unique tournament"
    unique_tournament_id: Unique tournament id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/unique-tournaments/featured-events"
    querystring = {'unique_tournament_id': unique_tournament_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_transfers(page: int, position_type: str='G', sort_type: str='transfer_fee_max', popularity: int=10, min_age: int=15, max_age: int=50, joined: bool=None, nationality: str='ENG', unique_tournament_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search transfers by parameters"
    page: page
        position_type: position_type: C, D, F, G, M
        sort_type: transfer_fee_max, transfer_fee_min, popularity_min, popularity_max, transfer_date_min, transfer_date_max
        popularity: popularity
        min_age: min_age
        max_age: max_age
        joined: joined  or no
        nationality: nationality
        unique_tournament_id: unique_tournament_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/transfers/search"
    querystring = {'page': page, }
    if position_type:
        querystring['position_type'] = position_type
    if sort_type:
        querystring['sort_type'] = sort_type
    if popularity:
        querystring['popularity'] = popularity
    if min_age:
        querystring['min_age'] = min_age
    if max_age:
        querystring['max_age'] = max_age
    if joined:
        querystring['joined'] = joined
    if nationality:
        querystring['nationality'] = nationality
    if unique_tournament_id:
        querystring['unique_tournament_id'] = unique_tournament_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sports_number_live_events(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get number of live events in the sports"
    
    """
    url = f"https://sofasport.p.rapidapi.com/v1/sports/number-live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sport_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of sports"
    
    """
    url = f"https://sofasport.p.rapidapi.com/v1/sports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manager_events(page: int, course_events: str, manager_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get events of the manager"
    page: page
        manager_id: manager_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/managers/events"
    querystring = {'page': page, 'course_events': course_events, 'manager_id': manager_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manager_photo(manager_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager photo by manager ID"
    manager_id: manager_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/managers/photo"
    querystring = {'manager_id': manager_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def h2h_events(custom_event_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get h2h events by custom_event_id"
    custom_event_id: Custom event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/h2h-events"
    querystring = {'custom_event_id': custom_event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unique_seasons_list_motorsport(unique_stage_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of the unique stages by category_id (motorsport)"
    unique_stage_id: Unique stage id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/unique-stages/seasons"
    querystring = {'unique_stage_id': unique_stage_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def streaks(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team streaks by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/streaks"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tweets(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tweets by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/tweets"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lineups_esports(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get games (esports) by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/esports-lineups"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def predict(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get predict by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/predict"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def point_by_point(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get point by point data by event_id. (tennis)"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/point-by-point"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def heatmap(team_id: int, event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get heatmap player by event_id and team_id"
    team_id: Team id
        event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/heatmap"
    querystring = {'team_id': team_id, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sub_events(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sub events by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/sub-events"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def media(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get media by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/media"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def incidents(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get incidents by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/incidents"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schedule_by_category(date: str, category_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event schedule by category and date"
    date: Date
        category_id: Category id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/schedule/category"
    querystring = {'date': date, 'category_id': category_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def graph_points(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get graph points by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/graph-points"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_data(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event data by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/data"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schedule_by_date(sport_id: int, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event schedule by date"
    sport_id: Sport id
        date: Date
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/schedule/date"
    querystring = {'sport_id': sport_id, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goal_distributions(season_id: int, unique_tournament_id: int, team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get goal distributions by team ID, unique tournament ID, season ID"
    season_id: season_id
        unique_tournament_id: unique_tournament_id
        team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/goal-distributions"
    querystring = {'season_id': season_id, 'unique_tournament_id': unique_tournament_id, 'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stage_seasons_motorsport(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team stage seasons (motorsport)"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/stage-seasons"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistics(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/statistics"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons_standings(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get seasons of the standings by team ID"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/standings/seasons"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons_for_team_statistics(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get seasons for team statistics by team ID"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/statistics/seasons"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def career_history_motorsport(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team career history (motorsport)"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/career-history"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_of_team(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players of team by team ID"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/players"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfers_of_team(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get transfers of team by team ID"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/transfers"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_latest_media(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest media by team ID. Ex. Full Highlights"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/latest-media"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_recent_form(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team recent form by team ID"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/recent-form"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_near_events(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team near events by team ID"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/near-events"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_data(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team data by team ID"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/data"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistics_esports(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics (esports) by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/esports-statistics"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def current_team_tournaments(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current team tournaments by team ID"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/current-tournaments"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_events(team_id: int, page: int, course_events: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get events by team ID"
    team_id: team_id
        page: page
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/events"
    querystring = {'team_id': team_id, 'page': page, 'course_events': course_events, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recent_unique_tournaments(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent unique tournaments by team ID"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/recent-unique-tournaments"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schedule_odds(date: str, odds_format: str, sport_id: int, provider_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get events schedule odds by date"
    date: Date
        sport_id: Sport id
        provider_id: Provider id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/schedule/odds"
    querystring = {'date': date, 'odds_format': odds_format, 'sport_id': sport_id, }
    if provider_id:
        querystring['provider_id'] = provider_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistics_overall(team_id: int, unique_tournament_id: int, season_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team overall statistics by team ID, unique tournament ID, season ID"
    team_id: team_id
        unique_tournament_id: unique_tournament_id
        season_id: season_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/statistics/result"
    querystring = {'team_id': team_id, 'unique_tournament_id': unique_tournament_id, 'season_id': season_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def popular_events(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get today's popular events"
    
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/schedule/popular"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_logo(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team logo by team ID"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/logo"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channels_data_stage(channel_id: int, stage_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get channels data (stage)"
    channel_id: Channel id
        stage_id: Stage id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/tv/stage-channel-data"
    querystring = {'channel_id': channel_id, 'stage_id': stage_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channels_list(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of channels by country"
    
    """
    url = f"https://sofasport.p.rapidapi.com/v1/tv/channels"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_player_statistics(season_id: int, unique_tournament_id: int, team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get overall player statistics by team ID, unique tournament ID, season ID"
    season_id: season_id
        unique_tournament_id: unique_tournament_id
        team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/player-statistics/result"
    querystring = {'season_id': season_id, 'unique_tournament_id': unique_tournament_id, 'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lineups(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get lineups by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/lineups"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons_for_player_statistics(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get seasons for player statistics by team ID"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/player-statistics/seasons"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_channels(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tv country channels by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/tv/country-channels"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_feed(team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news feed of the team"
    team_id: team_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/teams/news-feed"
    querystring = {'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def category_list(sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of categories"
    sport_id: Sport id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/categories"
    querystring = {'sport_id': sport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unique_tournaments_top_list(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a top list of unique tournaments by category id"
    
    """
    url = f"https://sofasport.p.rapidapi.com/v1/unique-tournaments-top"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tournament_seasons(tournament_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get seasons of the tournament"
    tournament_id: Tournament id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/tournaments/seasons"
    querystring = {'tournament_id': tournament_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def multi_search(group: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Multi search. Search by teams, referees, managers, players, tournaments"
    query: Query
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/search/multi"
    querystring = {'group': group, 'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manager_career_history(manager_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the manager's career history"
    manager_id: manager_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/managers/career-history"
    querystring = {'manager_id': manager_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def suggest(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Suggest search. Search Suggest automatically recommends popular searches as you type your query into the search field"
    query: Query
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/search/suggest"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_heatmap(seasons_id: int, unique_tournament_id: int, player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a player's heatmap"
    seasons_id: seasons_id
        unique_tournament_id: unique_tournament_id
        player_id: player_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/players/heatmap"
    querystring = {'seasons_id': seasons_id, 'unique_tournament_id': unique_tournament_id, 'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def referee_statistics(referee_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get referee statistics by referee ID"
    referee_id: referee_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/referees/statistics"
    querystring = {'referee_id': referee_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transfer_history(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player transfer history"
    player_id: player_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/players/transfer-history"
    querystring = {'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def attribute_overviews(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player attribute overviews"
    player_id: player_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/players/attribute-overviews"
    querystring = {'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_latest_media(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player's latest media by player_id"
    player_id: player_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/players/media"
    querystring = {'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_events(page: int, player_id: int, course_events: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get events by player ID"
    page: page
        player_id: player_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/players/events"
    querystring = {'page': page, 'player_id': player_id, 'course_events': course_events, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def last_year_summary(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get last year summary of the player"
    player_id: player_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/players/last-year-summary"
    querystring = {'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_statistics_seasons(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get seasons of the player"
    player_id: player_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/players/statistics/seasons"
    querystring = {'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_data(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player data by player ID"
    player_id: player_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/players/data"
    querystring = {'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def daily_unique_tournaments(timezone: int, month: str, sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily unique tournaments by date and sport"
    timezone: Timezone
        month: Month
        sport_id: Sport id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/calendar/daily-unique-tournaments"
    querystring = {'timezone': timezone, 'month': month, 'sport_id': sport_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def referee_data(referee_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get referee data by referee ID"
    referee_id: referee_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/referees/data"
    querystring = {'referee_id': referee_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def national_team_statistics(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player's stats by national teams"
    player_id: player_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/players/national-team-statistics"
    querystring = {'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_characteristics(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player characteristics by player ID"
    player_id: player_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/players/characteristics"
    querystring = {'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def referee_events(page: int, referee_id: int, course_events: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get events by referee ID"
    page: page
        referee_id: referee_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/referees/events"
    querystring = {'page': page, 'referee_id': referee_id, 'course_events': course_events, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_statistics(seasons_id: int, player_stat_type: str, player_id: int, unique_tournament_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player statistics. Use `regularSeason` - for hockey"
    seasons_id: seasons_id
        player_id: player_id
        unique_tournament_id: unique_tournament_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/players/statistics/result"
    querystring = {'seasons_id': seasons_id, 'player_stat_type': player_stat_type, 'player_id': player_id, 'unique_tournament_id': unique_tournament_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manager_data(manager_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get manager data by manager ID"
    manager_id: manager_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/managers/data"
    querystring = {'manager_id': manager_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def last_ratings(player_id: int, unique_tournament_id: int, seasons_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest player rating"
    player_id: player_id
        unique_tournament_id: unique_tournament_id
        seasons_id: seasons_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/players/last-ratings"
    querystring = {'player_id': player_id, 'unique_tournament_id': unique_tournament_id, 'seasons_id': seasons_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_week_rounds(seasons_id: int, unique_tournament_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team-week rounds of the season"
    seasons_id: seasons_id
        unique_tournament_id: unique_tournament_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/seasons/team-week/rounds"
    querystring = {'seasons_id': seasons_id, 'unique_tournament_id': unique_tournament_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories(sport_id: int, timezone: int, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get actual categories by date and sport"
    sport_id: Sport id
        timezone: Timezone
        date: Date
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/calendar/categories"
    querystring = {'sport_id': sport_id, 'timezone': timezone, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cup_trees(seasons_id: int, unique_tournament_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get cup trees data"
    seasons_id: seasons_id
        unique_tournament_id: unique_tournament_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/seasons/cup-trees"
    querystring = {'seasons_id': seasons_id, 'unique_tournament_id': unique_tournament_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_events(seasons_id: int, unique_tournament_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team event of the season"
    seasons_id: seasons_id
        unique_tournament_id: unique_tournament_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/seasons/team-events"
    querystring = {'seasons_id': seasons_id, 'unique_tournament_id': unique_tournament_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events(page: int, course_events: str, unique_tournament_id: int, seasons_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get events of the season"
    page: page
        unique_tournament_id: unique_tournament_id
        seasons_id: seasons_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/seasons/events"
    querystring = {'page': page, 'course_events': course_events, 'unique_tournament_id': unique_tournament_id, 'seasons_id': seasons_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_translations(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Teams translations"
    
    """
    url = f"https://sofasport.p.rapidapi.com/v1/translations/teams"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rounds(seasons_id: int, unique_tournament_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rounds of the season"
    seasons_id: seasons_id
        unique_tournament_id: unique_tournament_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/seasons/rounds"
    querystring = {'seasons_id': seasons_id, 'unique_tournament_id': unique_tournament_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings(standing_type: str, seasons_id: int, unique_tournament_id: int=17, tournament_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get standings of the season"
    seasons_id: seasons_id
        unique_tournament_id: unique_tournament_id
        tournament_id: tournament_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/seasons/standings"
    querystring = {'standing_type': standing_type, 'seasons_id': seasons_id, }
    if unique_tournament_id:
        querystring['unique_tournament_id'] = unique_tournament_id
    if tournament_id:
        querystring['tournament_id'] = tournament_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standing_competitor_motorsport(stage_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get standing competitor by stage_id (motorsport)"
    stage_id: Stage id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/stages/standings/competitor"
    querystring = {'stage_id': stage_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_statistics(seasons_statistics_type: str, unique_tournament_id: int, seasons_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get teams statistics of the season"
    unique_tournament_id: unique_tournament_id
        seasons_id: seasons_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/seasons/teams-statistics/result"
    querystring = {'seasons_statistics_type': seasons_statistics_type, 'unique_tournament_id': unique_tournament_id, 'seasons_id': seasons_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standing_teams_motorsport(stage_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get standing teams by stage_id (motorsport)"
    stage_id: Stage id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/stages/standings/teams"
    querystring = {'stage_id': stage_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_data(unique_tournament_id: int, seasons_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about the season"
    unique_tournament_id: unique_tournament_id
        seasons_id: seasons_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/seasons/data"
    querystring = {'unique_tournament_id': unique_tournament_id, 'seasons_id': seasons_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unique_stage_logo_motorsport(unique_stage_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get unique stage logo by unique_stage_id (motorsport)"
    unique_stage_id: Unique stage id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/unique-stages/logo"
    querystring = {'unique_stage_id': unique_stage_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_photo(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player photo by player ID"
    player_id: player_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/players/photo"
    querystring = {'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stage_data_motorsport(stage_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stage data by stage_id (motorsport)"
    stage_id: Stage id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/stages/data"
    querystring = {'stage_id': stage_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def substages_motorsport(stage_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get substages by stage_id (motorsport)"
    stage_id: Stage id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/stages/substages"
    querystring = {'stage_id': stage_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unique_stages_list_motorsport(category_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of the unique stages by category_id (motorsport)"
    category_id: category_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/unique-stages"
    querystring = {'category_id': category_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_player_statistics(event_id: int, player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player statistics by event_id"
    event_id: Event id
        player_id: Player id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/player-statistics"
    querystring = {'event_id': event_id, 'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stage_logo_motorsport(stage_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stage logo by stage_id (motorsport)"
    stage_id: Stage id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/stages/logo"
    querystring = {'stage_id': stage_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_statistics(unique_tournament_id: int, seasons_statistics_type: str, seasons_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players statistics of the season"
    unique_tournament_id: unique_tournament_id
        seasons_id: seasons_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/seasons/players-statistics/result"
    querystring = {'unique_tournament_id': unique_tournament_id, 'seasons_statistics_type': seasons_statistics_type, 'seasons_id': seasons_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds_winning(event_id: int, odds_format: str, provider_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get winning odds by event_id. You can convert them to decimals."
    event_id: Event id
        provider_id: Provider id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/odds/winning"
    querystring = {'event_id': event_id, 'odds_format': odds_format, }
    if provider_id:
        querystring['provider_id'] = provider_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_heatmap(player_id: int, event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get player heatmap by event_id"
    player_id: Player id
        event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/player-heatmap"
    querystring = {'player_id': player_id, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_week_players(round_id: int, seasons_id: int, unique_tournament_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team-week players of the season"
    round_id: round_id
        seasons_id: seasons_id
        unique_tournament_id: unique_tournament_id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/seasons/team-week/result"
    querystring = {'round_id': round_id, 'seasons_id': seasons_id, 'unique_tournament_id': unique_tournament_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def h2h_stats(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get H2H stats by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/h2h-stats"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_shirts(team_player: str, team: str, event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team shirts by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/team-shirts"
    querystring = {'team_player': team_player, 'team': team, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rounds_esports(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rounds (esports) by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/esports-rounds"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bans_esports(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get bans (esports) by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/esports-bans"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games_esports(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get games (esports) by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/esports-games"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def best_players(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get best players by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/best-players"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def form(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pregame team form by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/form"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def newly_added_events(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Newly added events"
    
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/newly-added-events"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def odds_all(event_id: int, odds_format: str, provider_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all odds by event_id. You can convert them to decimals. `provider_id`: 1 - Bet365, 100 - betano, 101 - bilyoner"
    event_id: Event id
        provider_id: Provider id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/odds/all"
    querystring = {'event_id': event_id, 'odds_format': odds_format, }
    if provider_id:
        querystring['provider_id'] = provider_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shotmap(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get shotmap by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/shotmap"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def event_managers(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team managers by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/managers"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fan_rating(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fan rating by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/fan-rating"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def innings_cricket(event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get innings (cricket) by event_id"
    event_id: Event id
        
    """
    url = f"https://sofasport.p.rapidapi.com/v1/events/innings"
    querystring = {'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sofasport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

