import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def team_transfers(season_id: str, t: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get transfers by team ID"
    season_id: Season ID
        t: Type
        id: Team ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/teams"
    querystring = {'season_id': season_id, 't': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_trend_analyses(is_id: int, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Match Stats Trend Analyses by Match ID"
    id: Match ID
        t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/stats"
    querystring = {'id': is_id, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_by_season_date(t: str, is_id: int, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Fixtures by Season ID and Date"
    t: Type
        id: Season ID
        d: Date
        page: Pagination
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'t': t, 'id': is_id, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_by_date_round(t: str, round_id: int, d: str='2019-12-21', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Fixtures Date and Round ID"
    t: Type
        round_id: Round ID
        d: Date
        page: Pagination
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'t': t, 'round_id': round_id, }
    if d:
        querystring['d'] = d
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_by_date_stage(t: str, stage_id: int, p: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Fixtures by Date and Stage ID"
    t: Type
        stage_id: Stage ID
        d: Date
        p: Pagination
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'t': t, 'stage_id': stage_id, }
    if p:
        querystring['p'] = p
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_by_season(is_id: str, t: str, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Fixtures by Season ID"
    id: Season ID
        t: Type
        page: Pagination
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'id': is_id, 't': t, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_by_date_season(t: str, d: str, season_id: str='16074', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Fixtures by Date and Season ID"
    t: Type
        d: Date
        season_id: Season ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'t': t, 'd': d, }
    if season_id:
        querystring['season_id'] = season_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_by_round(is_id: int, t: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Fixtures by Round ID"
    id: Round ID
        t: Type
        page: Pagination
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'id': is_id, 't': t, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_by_stage(t: str, stage_id: int, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Fixtures by Stage ID"
    t: Type
        stage_id: Stage ID
        page: Pagination
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'t': t, 'stage_id': stage_id, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stage_by_id(t: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Stage ID"
    t: Type
        id: Stage ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/stages"
    querystring = {'t': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_by_season_team(season_id: int, t: str, team_id: int, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Fixtures by Season ID and Team ID"
    season_id: Team ID
        t: Type
        team_id: Season ID
        page: Pagination
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'season_id': season_id, 't': t, 'team_id': team_id, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_countries(t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Countries endpoint gives you Country data like its Name, Flag, IsoCode, Continent and other related Country data.                         The Countries endpoint also gives the chance to order Leagues under the same Country. On the off chance that for instance, you need to demonstrate the Leagues per Country on your site/application, this may be a valuable endpoint for you by use the Leagues Endpoint."
    
    """
    url = f"https://soccersapi.p.rapidapi.com/countries"
    querystring = {'t': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_by_id(t: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Country by ID"
    t: Type
        id: Country ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/countries"
    querystring = {'t': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_leagues(t: str, page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Leagues endpoint would return a response with all Leagues available in your plan you are subscribed. The Leagues endpoint give you League data for example ID, Name, Country, Coverage etc. The Leagues endpoint also is used in combination with the Season or Seasons Endpoints. The Seasons Endpoint show information about all seasons available for the specific Leagues."
    
    """
    url = f"https://soccersapi.p.rapidapi.com/leagues"
    querystring = {'t': t, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_by_id(is_id: int, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get League by ID"
    id: League ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/leagues"
    querystring = {'id': is_id, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def standings(t: str, season_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Standings by Season ID"
    season_id: Season ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/leagues"
    querystring = {'t': t, 'season_id': season_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_by_id(is_id: int, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Season ID"
    id: Season ID
        t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/seasons"
    querystring = {'id': is_id, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rounds_by_season(t: str, season_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of Rounds by Season ID"
    t: Type
        season_id: Season ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/rounds"
    querystring = {'t': t, 'season_id': season_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def round_by_id(t: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Round by ID"
    t: type
        id: Round ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/rounds"
    querystring = {'t': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_by_season(season_id: int, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get participant Teams by Season ID"
    season_id: Season ID
        t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/teams"
    querystring = {'season_id': season_id, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_by_id(is_id: int, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Team by ID"
    id: Team ID
        t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/teams"
    querystring = {'id': is_id, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_squad(season_id: int, t: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get squad by team ID"
    season_id: Season ID
        t: Type
        id: Team ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/teams"
    querystring = {'season_id': season_id, 't': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def today(t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Today Matches"
    t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/livescores"
    querystring = {'t': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tomorrow(t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Tomorrow Matches"
    t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/livescores"
    querystring = {'t': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def yesterday(t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Yesterday Matches"
    t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/livescores"
    querystring = {'t': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def inplay(t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Inplay Matches"
    t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/livescores"
    querystring = {'t': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def not_started(t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Not Started (Today) Matches"
    t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/livescores"
    querystring = {'t': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ended(t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all Ended (Today) Matches"
    t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/livescores"
    querystring = {'t': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_by_date(d: str, t: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all matches of a specific date"
    d: Date
        t: Type
        page: Pagination
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'d': d, 't': t, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_leagues_by_country_id(country_id: int, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Get All Leagues Endpoint gives you all leagues from a specific country"
    
    """
    url = f"https://soccersapi.p.rapidapi.com/leagues"
    querystring = {'country_id': country_id, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixtures_by_date_country(is_id: int, page: str='1', t: str='schedule', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Fixtures by Date and Country ID"
    id: Country ID
        d: Date
        page: Pagination
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'id': is_id, }
    if page:
        querystring['page'] = page
    if t:
        querystring['t'] = t
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_by_id(t: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get match by ID"
    t: Type
        id: Match ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'t': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def commentaries(t: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get commentaries by ID"
    t: Type
        id: Match ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'t': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events(t: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get events by Fixture ID"
    t: Type
        id: Match ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'t': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def lineups(t: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get lineups by match ID"
    t: Type
        id: Match ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'t': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bench(t: str, is_id: str='15006543', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Bench by Fixture ID"
    t: Type
        is_id: Match ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'t': t, }
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def match_stats(is_id: int, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Statistics By Match ID"
    id: Match ID
        t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/stats"
    querystring = {'id': is_id, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_stats_current_season(is_id: str, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get League Stats by League ID"
    id: League ID
        t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/stats"
    querystring = {'id': is_id, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_stats(is_id: str, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get League Stats by Season ID"
    id: Season ID
        t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/stats"
    querystring = {'id': is_id, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_stats(is_id: str, season_id: str, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Stats By Team ID and Season ID"
    id: Team ID
        season_id: Season ID
        t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/stats"
    querystring = {'id': is_id, 'season_id': season_id, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_stats(season_id: str, t: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Statistics by Player ID"
    season_id: Season ID
        t: Type
        id: Player ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/stats"
    querystring = {'season_id': season_id, 't': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topscorers(t: str, season_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Topscorers by Season ID"
    
    """
    url = f"https://soccersapi.p.rapidapi.com/topscorers"
    querystring = {'t': t, 'season_id': season_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bookmakers(t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of Bookmakers"
    t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/bookmakers"
    querystring = {'t': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bookmaker_by_id(is_id: int, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Bookmaker by ID"
    id: Bookmaker ID
        t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/bookmakers"
    querystring = {'id': is_id, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def markets(t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of Markets"
    t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/markets"
    querystring = {'t': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_by_id(t: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Market by ID"
    t: Type
        id: Market ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/markets"
    querystring = {'t': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pre_match_odds(t: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Pre-Match Odds by Fixture ID"
    t: Type
        id: Match ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'t': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def in_play_match_odds(t: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Inplay Odds by Match ID"
    t: Type
        id: Match ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'t': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_by_id(t: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Player ID"
    t: Type
        id: Player ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/players"
    querystring = {'t': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_by_country(t: str, country_id: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of players from a single country by country ID"
    t: Type
        country_id: Country ID
        page: Pagination
        
    """
    url = f"https://soccersapi.p.rapidapi.com/players"
    querystring = {'t': t, 'country_id': country_id, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tv_coverage(t: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get international TV channels coverage by match ID"
    t: Type
        id: Match ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/fixtures"
    querystring = {'t': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coach_by_id(is_id: int, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Coach by ID"
    id: Coach ID
        t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/coaches"
    querystring = {'id': is_id, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venue_by_id(is_id: int, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Venues by ID"
    id: Venue ID
        t: Type
        
    """
    url = f"https://soccersapi.p.rapidapi.com/venues"
    querystring = {'id': is_id, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def referee_info(t: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Referees by ID"
    id: Referee ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/referees"
    querystring = {'t': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def teams_by_country(country_id: str, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Teams by Country ID endpoint returns a list of teams from a specific country"
    
    """
    url = f"https://soccersapi.p.rapidapi.com/teams"
    querystring = {'country_id': country_id, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topassists(season_id: str, t: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Topassists by Season ID"
    
    """
    url = f"https://soccersapi.p.rapidapi.com/topscorers"
    querystring = {'season_id': season_id, 't': t, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topcards(t: str, season_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Topcards by Season ID"
    
    """
    url = f"https://soccersapi.p.rapidapi.com/topscorers"
    querystring = {'t': t, 'season_id': season_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_sidelined(t: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sidelined by team ID"
    t: Type
        id: Team ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/teams"
    querystring = {'t': t, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_trophies(t: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get team trophies"
    t: Type
        id: Team ID
        
    """
    url = f"https://soccersapi.p.rapidapi.com/teams"
    querystring = {'t': t, 'id': is_id, }
    if t:
        querystring['t'] = t
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "soccersapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

