import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def soccer_news_detail(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news detail by news `slug`"
    slug: Get `slug` from `/soccer/news-list` endpoint
        
    """
    url = f"https://livescore-football.p.rapidapi.com/soccer/news-detail"
    querystring = {'slug': slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soccer_league_table(country_code: str, league_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current league table standings by `league_code` and `country_code`"
    country_code: Country Code including federations. Get from `/soccer/countries` or `/leagues-by-country` endpoint
        league_code: League Code. Get data from `/soccer/leagues-by-country` endpoint
        
    """
    url = f"https://livescore-football.p.rapidapi.com/soccer/league-table"
    querystring = {'country_code': country_code, 'league_code': league_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soccer_leagues_by_country(country_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all leagues in a country by `country_code`"
    country_code: Country code or Federation code. Get data from `/soccer/countries` endpoint
        
    """
    url = f"https://livescore-football.p.rapidapi.com/soccer/leagues-by-country"
    querystring = {'country_code': country_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soccer_live_matches(timezone_utc: str='7:00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all matches by league"
    timezone_utc: Desired timezone. Get options from `/soccer/timezones` endpoint. Default value `0:00`
        
    """
    url = f"https://livescore-football.p.rapidapi.com/soccer/live-matches"
    querystring = {}
    if timezone_utc:
        querystring['timezone_utc'] = timezone_utc
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soccer_match_commentaries(match_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match tracker / commentaries by `match_id`"
    match_id: Get `match_id` from `/soccer/matches-by-league` or `/soccer/matches-by-date` endpoint
        
    """
    url = f"https://livescore-football.p.rapidapi.com/soccer/match-commentaries"
    querystring = {'match_id': match_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soccer_match_events(match_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match events / incidents during a match by `match_id`"
    match_id: Get `match_id` from `/soccer/matches-by-league` or `/soccer/matches-by-date` endpoint
        
    """
    url = f"https://livescore-football.p.rapidapi.com/soccer/match-events"
    querystring = {'match_id': match_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soccer_match_h2h(match_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team_1 & team_2 latest matches and Head to latest Head matches by `match_id`"
    match_id: Get `match_id` from `/soccer/matches-by-league` or `/soccer/matches-by-date` endpoint
        
    """
    url = f"https://livescore-football.p.rapidapi.com/soccer/match-h2h"
    querystring = {'match_id': match_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soccer_match_lineups(match_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match line-ups for team_1 and team_2 by `match_id`"
    match_id: Get `match_id` from `/soccer/matches-by-league` or `/soccer/matches-by-date` endpoint
        
    """
    url = f"https://livescore-football.p.rapidapi.com/soccer/match-lineups"
    querystring = {'match_id': match_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soccer_match_statistics(match_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a match's statistics by `match_id`"
    match_id: Get `match_id` from `/matches-by-league` or `/soccer/matches-by-date` endpoint
        
    """
    url = f"https://livescore-football.p.rapidapi.com/soccer/match-statistics"
    querystring = {'match_id': match_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soccer_matches_by_date(date: int, league_code: str='premier-league', timezone_utc: str='0:00', country_code: str='england', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all matches by date. Filtered by `country_code` and `league_code`"
    date: Date string with following format `ddddmmyy`
        league_code: League Code. Get data from `/soccer/leagues-by-country` endpoint
        timezone_utc: Desired timezone. Get options from `/soccer/timezones` endpoint. Default value `0:00`
        country_code: Country Code including federations. Get from `/soccer/countries` or `/leagues-by-country` endpoint
        
    """
    url = f"https://livescore-football.p.rapidapi.com/soccer/matches-by-date"
    querystring = {'date': date, }
    if league_code:
        querystring['league_code'] = league_code
    if timezone_utc:
        querystring['timezone_utc'] = timezone_utc
    if country_code:
        querystring['country_code'] = country_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soccer_matches_by_league(league_code: str, country_code: str, timezone_utc: str='0:00', round: str='1/16', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all matches by league"
    league_code: League Code. Get data from `/soccer/leagues-by-country` endpoint
        country_code: Country Code including federations. Get from `/soccer/countries` or `/leagues-by-country` endpoint
        timezone_utc: Desired timezone. Get options from `/soccer/timezones` endpoint. Default value `0:00`
        round: Round can be number or string e.g: 1|8|1/16 etc.
        
    """
    url = f"https://livescore-football.p.rapidapi.com/soccer/matches-by-league"
    querystring = {'league_code': league_code, 'country_code': country_code, }
    if timezone_utc:
        querystring['timezone_utc'] = timezone_utc
    if round:
        querystring['round'] = round
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soccer_news_list(page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news list from livescore soccer news"
    
    """
    url = f"https://livescore-football.p.rapidapi.com/soccer/news-list"
    querystring = {'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soccer_timezones(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get timezones"
    
    """
    url = f"https://livescore-football.p.rapidapi.com/soccer/timezones"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def soccer_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all countries & federations"
    
    """
    url = f"https://livescore-football.p.rapidapi.com/soccer/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "livescore-football.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

