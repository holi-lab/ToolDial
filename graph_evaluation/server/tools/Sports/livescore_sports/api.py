import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def media_details(timezone: int, locale: str, media_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Media details (video playlist). Ex media_id in /v1/competitions/details"
    timezone: Timezone, offsets from UTC
        media_id: Media ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/media/details"
    querystring = {'timezone': timezone, 'locale': locale, 'media_id': media_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_live_list(locale: str, timezone: int, sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the live events by sport"
    timezone: Timezone, offsets from UTC
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/events/live"
    querystring = {'locale': locale, 'timezone': timezone, 'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def media_watch(locale: str, timezone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Media watch (video playlist)"
    timezone: Timezone, offsets from UTC
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/media/watch"
    querystring = {'locale': locale, 'timezone': timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def populars_items(locale: str, popular_category: str, sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the populars items"
    
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/populars/items"
    querystring = {'locale': locale, 'popular_category': popular_category, 'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_media(sport: str, locale: str, event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event media form event ID"
    event_id: Event ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/events/media"
    querystring = {'sport': sport, 'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def images_team(badge_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team image"
    badge_id: Badge id, ex. `10260`. BADGE_ID field
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/images/team"
    querystring = {'badge_id': badge_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def images_category(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get category image"
    slug: Slug, ex. `france` `champions-league` `intl`
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/images/category"
    querystring = {'slug': slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_h2h(sport: str, event_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event H2H (head to head) form event ID"
    event_id: Event ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/events/h2h"
    querystring = {'sport': sport, 'event_id': event_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_statistics(competition_type: str, timezone: int, team_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team statistics by team ID."
    timezone: Timezone, offsets from UTC
        team_id: Team ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/teams/statistics"
    querystring = {'competition_type': competition_type, 'timezone': timezone, 'team_id': team_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_info(sport: str, locale: str, event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event info form event ID"
    event_id: Event ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/events/info"
    querystring = {'sport': sport, 'locale': locale, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_details(team_id: int, timezone: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team details by team ID. News tag, country, badge_id (logo), result and fixtures events"
    team_id: Team ID
        timezone: Timezone, offsets from UTC
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/teams/details"
    querystring = {'team_id': team_id, 'timezone': timezone, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_details_short(team_ids: str, timezone: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team short details by team IDs (list)"
    team_ids: Team IDs
        timezone: Timezone, offsets from UTC
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/teams/short-details"
    querystring = {'team_ids': team_ids, 'timezone': timezone, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_count_live(sport: str, locale: str, timezone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the count live events by sport"
    timezone: Timezone, offsets from UTC
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/events/count-live"
    querystring = {'sport': sport, 'locale': locale, 'timezone': timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_scoreboard(locale: str, sport: str, event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event scoreboard by event ID"
    event_id: Event ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/events/scoreboard"
    querystring = {'locale': locale, 'sport': sport, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_incidents(sport: str, event_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event incidents form event ID"
    event_id: Event ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/events/incidents"
    querystring = {'sport': sport, 'event_id': event_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_competition_standings(sport: str, event_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event competition standings (table) by event ID"
    event_id: Event ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/events/competition-standings"
    querystring = {'sport': sport, 'event_id': event_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_player_statistics(team_id: int, competition_id: int, locale: str, competition_type: str, stat_type: str, timezone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team player statistics by team ID. TYPE_OF_STATs: goals: '1', assists: '3',  red_cards: '4',  yellow_cards: '6', shots_on_target: '8',"
    team_id: Team ID
        competition_id: Competition ID
        timezone: Timezone, offsets from UTC
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/teams/player-statistics"
    querystring = {'team_id': team_id, 'competition_id': competition_id, 'locale': locale, 'competition_type': competition_type, 'stat_type': stat_type, 'timezone': timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_next_event(timezone: int, team_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team next event by team ID."
    timezone: Timezone, offsets from UTC
        team_id: Team ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/teams/next-events"
    querystring = {'timezone': timezone, 'team_id': team_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def meta_specification(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get meta specification data"
    
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/meta/specification"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def meta_sports_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the sport"
    
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/meta/sports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_list(date: str, locale: str, timezone: int, sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the events by sport and date"
    date: Date
        timezone: Timezone, offsets from UTC
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/events/list"
    querystring = {'date': date, 'locale': locale, 'timezone': timezone, 'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_category(category: str, locale: str, timezone: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "News category"
    timezone: Timezone, offsets from UTC
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/news/category"
    querystring = {'category': category, 'locale': locale, 'timezone': timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def meta_translations(locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get translations data"
    
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/meta/translations"
    querystring = {'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_comments(event_id: int, sport: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event comments form event ID"
    event_id: Event ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/events/comments"
    querystring = {'event_id': event_id, 'sport': sport, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_team_form(event_id: int, locale: str, sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event team form event ID"
    event_id: Event ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/events/form"
    querystring = {'event_id': event_id, 'locale': locale, 'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stages_list(sport: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the stages"
    
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/stages/list"
    querystring = {'sport': sport, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_media(locale: str, timezone: int, team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team media by team ID."
    timezone: Timezone, offsets from UTC
        team_id: Team ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/teams/media"
    querystring = {'locale': locale, 'timezone': timezone, 'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_standings_short(timezone: int, locale: str, team_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team standings by team ID."
    timezone: Timezone, offsets from UTC
        team_id: Team ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/teams/standings"
    querystring = {'timezone': timezone, 'locale': locale, 'team_id': team_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_player_stats(stat_type: str, competition_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition Player statistics by competition ID"
    competition_id: Competition ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/competitions/player-statistics"
    querystring = {'stat_type': stat_type, 'competition_id': competition_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_details(timezone: int, competition_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition details by competition ID"
    timezone: Timezone, offsets from UTC
        competition_id: Competition ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/competitions/details"
    querystring = {'timezone': timezone, 'competition_id': competition_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_standings_revs(timezone: int, locale: str, country_slug: str, stage_slug: str, sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition standings (table) by competition ID.  LTT_CODE: All(1),Home(2),Away 3"
    timezone: Timezone, offsets from UTC
        country_slug: Country slug
        stage_slug: Country slug
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/competitions/revs-standings"
    querystring = {'timezone': timezone, 'locale': locale, 'country_slug': country_slug, 'stage_slug': stage_slug, 'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_details(slug: str, timezone: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "News details"
    slug: Slug
        timezone: Timezone, offsets from UTC
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/news/details"
    querystring = {'slug': slug, 'timezone': timezone, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_statistics(event_id: int, sport: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event statistics form event ID"
    event_id: Event ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/events/statistics"
    querystring = {'event_id': event_id, 'sport': sport, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_lineups(locale: str, sport: str, event_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get event lineups form event ID"
    event_id: Event ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/events/lineups"
    querystring = {'locale': locale, 'sport': sport, 'event_id': event_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stages_events(stage_slug: str, locale: str, timezone: int, sport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the events for stage"
    stage_slug: Stage slug, use `Stages List`
        timezone: Timezone, offsets from UTC
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/stages/events"
    querystring = {'stage_slug': stage_slug, 'locale': locale, 'timezone': timezone, 'sport': sport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_standings(timezone: int, competition_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition standings (table) by competition ID.  LTT_CODE: All(1),Home(2),Away 3"
    timezone: Timezone, offsets from UTC
        competition_id: Competition ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/competitions/standings"
    querystring = {'timezone': timezone, 'competition_id': competition_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def competitions_team_stats(stat_type: str, competition_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get competition Team statistics by competition ID"
    competition_id: Competition ID
        
    """
    url = f"https://livescore-sports.p.rapidapi.com/v1/competitions/team-statistics"
    querystring = {'stat_type': stat_type, 'competition_id': competition_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-sports.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

