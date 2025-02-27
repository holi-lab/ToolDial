import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def in_game_ads(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All currently active Maniapubs displayed in-game."
    
    """
    url = f"https://trackmania.p.rapidapi.com/ads"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_cup_of_the_day_information(player_id: str, limit: int=50, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the COTD info of a player."
    player_id: The Account ID of the Player. Can be obtained using the `/players` endpoints.
        limit: The result limit that gets returned. Default limit is `50`.
        page: The result page, each containing 50 results. Default page is `0`.
        
    """
    url = f"https://trackmania.p.rapidapi.com/player/cotd"
    querystring = {'player_id': player_id, }
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_trophy_information(player_id: str, limit: int=50, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the trophy info of a player."
    player_id: The Account ID of the Player. Can be obtained using the `/players` endpoints.
        limit: The result limit that gets returned. Default limit is `50`.
        page: The result page, each containing 50 results. Default page is `0`.
        
    """
    url = f"https://trackmania.p.rapidapi.com/player/trophies"
    querystring = {'player_id': player_id, }
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_royal_information(player_id: str, limit: int=50, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the royal info of a player."
    player_id: The Account ID of the Player. Can be obtained using the `/players` endpoints.
        limit: The result limit that gets returned. Default limit is `50`.
        page: The result page, each containing 50 results. Default page is `0`.
        
    """
    url = f"https://trackmania.p.rapidapi.com/player/royal"
    querystring = {'player_id': player_id, }
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_matchmaking_information(player_id: str, page: int=0, limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the matchmaking info of a player."
    player_id: The Account ID of the Player. Can be obtained using the `/players` endpoints.
        page: The result page, each containing 50 results. Default page is `0`.
        limit: The result limit that gets returned. Default limit is `50`.
        
    """
    url = f"https://trackmania.p.rapidapi.com/player/matchmaking"
    querystring = {'player_id': player_id, }
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_summary(player_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the summary of a player containing general information."
    player_id: The Account ID of the Player. Can be obtained using the `/players` endpoints.
        
    """
    url = f"https://trackmania.p.rapidapi.com/player/summary"
    querystring = {'player_id': player_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sponsor_players(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Players that are sponsors."
    
    """
    url = f"https://trackmania.p.rapidapi.com/players/sponsor"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_all_players(search_query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for players with `search_query`"
    search_query: Search query used in the search. Has to be at least `4` characters.
        
    """
    url = f"https://trackmania.p.rapidapi.com/players/search"
    querystring = {'search_query': search_query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tmgl_players(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Players from tmgl."
    
    """
    url = f"https://trackmania.p.rapidapi.com/players/tmgl"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nadeo_players(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Players from nadeo."
    
    """
    url = f"https://trackmania.p.rapidapi.com/players/nadeo"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_players(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Players that are in the Openplanet Team."
    
    """
    url = f"https://trackmania.p.rapidapi.com/players/team"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def totd_tracks_by_date(month: int, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the TOTD tracks by date containing the year and month."
    month: month of the date.
        year: year of the date.
        
    """
    url = f"https://trackmania.p.rapidapi.com/totd/date"
    querystring = {'month': month, 'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def totd_tracks_by_page(page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the TOTD tracks by page number of the `https://trackmania.io/#/totd` page."
    page: The result page, each containing 50 results. Default page is `0`.
        
    """
    url = f"https://trackmania.p.rapidapi.com/totd/page"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def royal_matches(page: int=0, limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent royal matches."
    page: The result page, each containing 50 results. Default page is `0`.
        limit: The result limit that gets returned. Default limit is `50`.
        
    """
    url = f"https://trackmania.p.rapidapi.com/matches/royal"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchmaking_matches(limit: int=50, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent matchmaking matches."
    limit: The result limit that gets returned. Default limit is `50`.
        page: The result page, each containing 50 results. Default page is `0`.
        
    """
    url = f"https://trackmania.p.rapidapi.com/matches/matchmaking"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_players_by_royal(page: int=0, limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players by their royal rank."
    page: The result page, each containing 50 results. Default page is `0`.
        limit: The result limit that gets returned. Default limit is `50`.
        
    """
    url = f"https://trackmania.p.rapidapi.com/top_players/royal"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_players_by_matchmaking(limit: int=50, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players by their matchmaking rank."
    limit: The result limit that gets returned. Default limit is `50`.
        page: The result page, each containing 50 results. Default page is `0`.
        
    """
    url = f"https://trackmania.p.rapidapi.com/top_players/matchmaking"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_players_by_trophies(page: int=0, limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the top players by their tophy count."
    page: The result page, each containing 50 results. Default page is `0`.
        limit: The result limit that gets returned. Default limit is `50`.
        
    """
    url = f"https://trackmania.p.rapidapi.com/top_players/trophies"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trackmania.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

