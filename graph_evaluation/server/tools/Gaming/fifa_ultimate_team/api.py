import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def specific_league(league_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access information relating to a specific league."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/leagues/{league_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_leagues(limit: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access information about all leagues in the database."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/leagues"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_clubs(limit: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access information about all clubs in the database."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/clubs"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_price_by_id(player_id: int, price_update: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access the price of the player queried by ID. Pass price_update as a query parameter to get an a realtime price. price_update will take more time to return a response."
    price_update: **Example**: 0 (false) or 1 (true). Choose if you want to force a price update when querying the price. Having this set to true will update the players price, but may take longer for the request to complete
        
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/players/{player_id}/price"
    querystring = {}
    if price_update:
        querystring['price_update'] = price_update
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_players_by_id(player_id: str, page: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access information about all players that have been added to the database since the ID you provide."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/players/latest/{player_id}"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_image_by_id(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access the image of the player queried by ID"
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/players/{player_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_by_id(player_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access information about a specific player in the database."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/players/{player_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_by_club(club_id: int, limit: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access information about all players from a specific club in the database."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/players/club/{club_id}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_by_league(league_id: str, page: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access information about all players from a specific league in the database."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/players/league/{league_id}"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players_by_nation(nation_id: str, page: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access information about all players from a specific nation in the database."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/players/nation/{nation_id}"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_players(limit: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access information about all players in the database."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/players"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_cards(limit: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access information about all cards in the database."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/cards"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def card_image(card_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the image of the card matching the ID queried."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/cards/{card_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specific_card(card_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access information relating to a specific card."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/cards/{card_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specific_nation(nation_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access information relating to a specific nation."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/nations/{nation_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nation_badge(nation_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the badge of the nation matching the ID queried."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/nations/{nation_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_nations(page: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access information about all nations in the database."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/nations"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_badge(league_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the badge of the league matching the ID queried."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/leagues/{league_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def club_badge(club_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the badge of the club matching the ID queried."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/clubs/{club_id}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specific_club(clud_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access information relating to a specific club."
    
    """
    url = f"https://fifa-ultimate-team.p.rapidapi.com/clubs/{clud_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fifa-ultimate-team.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

