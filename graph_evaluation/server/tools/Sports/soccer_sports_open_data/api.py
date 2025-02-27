import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def season_rounds(league_slug: str, season_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns all the rounds available in a specified season"
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}/seasons/{season_slug}/rounds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_details(league_slug: str, season_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this API you could retrieve season details for a specified league. In result set you could find standings and rounds (if available)"
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}/seasons/{season_slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_leagues(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this API you can retrieve all the leagues available on Soccer Sports Open Data"
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasons_available(league_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this API you could retrieve which are the seasons available for a specified league"
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_standings(league_slug: str, season_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this API you could retrieve season league standings for a specified league."
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}/seasons/{season_slug}/standings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_details(league_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this API you retrieve league details like identifier, slug, nation or federation"
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_standings_position(league_slug: str, season_slug: str, position: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this API you could retrieve data for a specific standings position. You've to add in request league_slug, season_slug and position"
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}/seasons/{season_slug}/standings/{position}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_specified_round(league_slug: str, season_slug: str, round_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns a specified round, with full details, for a specific season"
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}/seasons/{season_slug}/rounds/{round_slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def round_matches(league_slug: str, season_slug: str, round_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns matches available in a specified round. Every match has a full detail about events occurred."
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}/seasons/{season_slug}/rounds/{round_slug}/matches"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def round_specified_match(league_slug: str, season_slug: str, round_slug: str, match_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns full details information about a specified match in a round."
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}/seasons/{season_slug}/rounds/{round_slug}/matches/{match_slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_teams_available(league_slug: str, season_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns a list of teams available in a league during a season."
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}/seasons/{season_slug}/teams"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stadiums_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns a list of stadiums available."
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/stadiums"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_top_scorer(league_slug: str, season_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this API you're able to get top scorer of a specific league during a season"
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}/seasons/{season_slug}/topscorers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_referee_in_season(league_slug: str, season_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns a list of available referees in a season for a specified league."
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}/seasons/{season_slug}/referees"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_managers_in_a_season_for_a_team(team_slug: str, league_slug: str, season_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns a list of managers for a specified team available in a league for a specified season."
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}/seasons/{season_slug}/teams/{team_slug}/managers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_teams_available_players(league_slug: str, season_slug: str, team_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this API you can get all the player roster for a specified team in a league during a season."
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}/seasons/{season_slug}/teams/{team_slug}/players"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_matches_for_a_team(league_slug: str, season_slug: str, team_identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Matches list for a team by searching it with Team Identifier (e.g. for Real Madrid is fw3ok0fty95tz0ydspi2g5yzghm5exdj)"
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}/seasons/{season_slug}/rounds?team_identifier={team_identifier}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def season_head_2_head(league_slug: str, season_slug: str, team_1: str, team_2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Head 2 Head between two teams during a season"
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/leagues/{league_slug}/seasons/{season_slug}/rounds?team_1_slug={team_1}&team_2_slug={team_2}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of Events available"
    
    """
    url = f"https://sportsop-soccer-sports-open-data-v1.p.rapidapi.com/v1/utils/events"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sportsop-soccer-sports-open-data-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

