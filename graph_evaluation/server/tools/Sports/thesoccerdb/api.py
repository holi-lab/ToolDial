import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_leagues(name: str=None, type: str=None, include: str=None, is_cup: bool=None, country_ids: str=None, per_page: int=100, page: int=1, ids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get leagues"
    name: Filter by `name`.
        type: Filter by `type`.
        include: Enriching the response. Allow multiple values, separated by `,`.
Available options: `country`, `season`, `seasons`
        is_cup: Filter by `is_cup`.
        country_ids: Filter by `country_id`. Allow multiple values, separated by `,`.
        ids: Filter by `id`. Allow multiple values, separated by `,`.
        
    """
    url = f"https://thesoccerdb.p.rapidapi.com/v1/leagues"
    querystring = {}
    if name:
        querystring['name'] = name
    if type:
        querystring['type'] = type
    if include:
        querystring['include'] = include
    if is_cup:
        querystring['is_cup'] = is_cup
    if country_ids:
        querystring['country_ids'] = country_ids
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    if ids:
        querystring['ids'] = ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thesoccerdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_seasons(is_current_season: bool=None, include: str=None, per_page: int=100, page: int=1, league_ids: str=None, ids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get seasons"
    is_current_season: Filter by `is_current_season`
        include: Enriching the response. Allow multiple values, separated by `,`.
Available options: `league`, `stages`, `rounds`
        league_ids: Filter by `league_id`. Allow multiple values, separated by `,`.
        ids: Filter by `id`. Allow multiple values, separated by `,`.
        
    """
    url = f"https://thesoccerdb.p.rapidapi.com/v1/seasons"
    querystring = {}
    if is_current_season:
        querystring['is_current_season'] = is_current_season
    if include:
        querystring['include'] = include
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    if league_ids:
        querystring['league_ids'] = league_ids
    if ids:
        querystring['ids'] = ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thesoccerdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_countries(name: str=None, per_page: int=100, ids: str=None, include: str=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get countries"
    name: Filter by `name`.
        ids: Filter by `id`. Allow multiple values, separated by `,`.
        include: Enriching the response. Allow multiple values, separated by `,`.
Available options: `leagues`
        
    """
    url = f"https://thesoccerdb.p.rapidapi.com/v1/countries"
    querystring = {}
    if name:
        querystring['name'] = name
    if per_page:
        querystring['per_page'] = per_page
    if ids:
        querystring['ids'] = ids
    if include:
        querystring['include'] = include
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thesoccerdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fixtures_matches(ids: str=None, visitorteam_ids: str=None, season_ids: str=None, league_ids: str=None, localteam_ids: str=None, round_ids: str=None, winner_team_ids: str=None, stage_ids: str=None, status: str=None, include: str=None, per_page: int=100, page: int=1, starting_time_lte: int=None, starting_time_gte: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fixtures (matches)"
    ids: Filter by `id`. Allow multiple values, separated by `,`.
        visitorteam_ids: Filter by `visitorteam_id`. Allow multiple values, separated by `,`.
        season_ids: Filter by `season_id`. Allow multiple values, separated by `,`.
        league_ids: Filter by `league_id`. Allow multiple values, separated by `,`.
        localteam_ids: Filter by `localteam_id`. Allow multiple values, separated by `,`.
        round_ids: Filter by `round_id`. Allow multiple values, separated by `,`.
        winner_team_ids: Filter by `winner_team_id`. Allow multiple values, separated by `,`.
        stage_ids: Filter by `stage_id`. Allow multiple values, separated by `,`.
        status: Filter by `time.status`.

Available options:

- `FT`: Full-time
- `NS`: Not started 
- `CANCL`: Cancelled
        include: Enriching the response. Allow multiple values, separated by `,`.
Available options: `league`, `season`, `stage`, `round`, `localteam`, `visitorteam`
        starting_time_lte: Filter by `starting_time.timestamp`, which value is less-or-equal.
The value should be in UNIX timestamp, and be converted to UTC.
        starting_time_gte: Filter by `starting_time.timestamp`, which value is greater-or-equal.
The value should be in UNIX timestamp, and be converted to UTC.
        
    """
    url = f"https://thesoccerdb.p.rapidapi.com/v1/fixtures"
    querystring = {}
    if ids:
        querystring['ids'] = ids
    if visitorteam_ids:
        querystring['visitorteam_ids'] = visitorteam_ids
    if season_ids:
        querystring['season_ids'] = season_ids
    if league_ids:
        querystring['league_ids'] = league_ids
    if localteam_ids:
        querystring['localteam_ids'] = localteam_ids
    if round_ids:
        querystring['round_ids'] = round_ids
    if winner_team_ids:
        querystring['winner_team_ids'] = winner_team_ids
    if stage_ids:
        querystring['stage_ids'] = stage_ids
    if status:
        querystring['status'] = status
    if include:
        querystring['include'] = include
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    if starting_time_lte:
        querystring['starting_time_lte'] = starting_time_lte
    if starting_time_gte:
        querystring['starting_time_gte'] = starting_time_gte
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thesoccerdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rounds(include: str=None, stage_ids: str=None, ids: str=None, league_ids: str=None, season_ids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rounds"
    include: Enriching the response. Allow multiple values, separated by `,`.
Available options: `league`, `season`, `stage`
        stage_ids: Filter by `stage_id`. Allow multiple values, separated by `,`.
        ids: Filter by `id`. Allow multiple values, separated by `,`.
        league_ids: Filter by `league_id`. Allow multiple values, separated by `,`.
        season_ids: Filter by `season_id`. Allow multiple values, separated by `,`.
        
    """
    url = f"https://thesoccerdb.p.rapidapi.com/v1/rounds"
    querystring = {}
    if include:
        querystring['include'] = include
    if stage_ids:
        querystring['stage_ids'] = stage_ids
    if ids:
        querystring['ids'] = ids
    if league_ids:
        querystring['league_ids'] = league_ids
    if season_ids:
        querystring['season_ids'] = season_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thesoccerdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_teams(include: str=None, per_page: int=100, ids: str=None, short_code: str=None, page: int=1, country_ids: str=None, name: str=None, is_national_team: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get teams"
    include: Enriching the response. Allow multiple values, separated by `,`.
Available options: `country`
        ids: Filter by `id`. Allow multiple values, separated by `,`.
        short_code: Filter by `short_code`
        country_ids: Filter by `country_id`. Allow multiple values, separated by `,`.
        name: Filter by `name`.
        is_national_team: Filter by `is_national_team`.
        
    """
    url = f"https://thesoccerdb.p.rapidapi.com/v1/teams"
    querystring = {}
    if include:
        querystring['include'] = include
    if per_page:
        querystring['per_page'] = per_page
    if ids:
        querystring['ids'] = ids
    if short_code:
        querystring['short_code'] = short_code
    if page:
        querystring['page'] = page
    if country_ids:
        querystring['country_ids'] = country_ids
    if name:
        querystring['name'] = name
    if is_national_team:
        querystring['is_national_team'] = is_national_team
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thesoccerdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stages(include: str=None, season_ids: str=None, ids: str=None, league_ids: str=None, per_page: int=100, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stages"
    include: Enriching the response. Allow multiple values, separated by `,`.
Available options: `league`, `season`, `rounds`
        season_ids: Filter by `season_id`. Allow multiple values, separated by `,`.
        ids: Filter by `id`. Allow multiple values, separated by `,`.
        league_ids: Filter by `league_id`. Allow multiple values, separated by `,`.
        
    """
    url = f"https://thesoccerdb.p.rapidapi.com/v1/stages"
    querystring = {}
    if include:
        querystring['include'] = include
    if season_ids:
        querystring['season_ids'] = season_ids
    if ids:
        querystring['ids'] = ids
    if league_ids:
        querystring['league_ids'] = league_ids
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thesoccerdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_players(ids: str=None, per_page: int=100, country_ids: str=None, page: int=1, include: str=None, display_name: str=None, team_ids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players"
    ids: Filter by `id`. Allow multiple values, separated by `,`.
        country_ids: Filter by `country_id`. Allow multiple values, separated by `,`.
        include: Enriching the response. Allow multiple values, separated by `,`.
Available options: `country`, `team`
        display_name: Filter by `display_name`.
        team_ids: Filter by `team_id`. Allow multiple values, separated by `,`.
        
    """
    url = f"https://thesoccerdb.p.rapidapi.com/v1/players"
    querystring = {}
    if ids:
        querystring['ids'] = ids
    if per_page:
        querystring['per_page'] = per_page
    if country_ids:
        querystring['country_ids'] = country_ids
    if page:
        querystring['page'] = page
    if include:
        querystring['include'] = include
    if display_name:
        querystring['display_name'] = display_name
    if team_ids:
        querystring['team_ids'] = team_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "thesoccerdb.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

