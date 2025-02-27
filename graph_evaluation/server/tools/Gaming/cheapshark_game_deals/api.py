import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stores_info(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a full list of store IDs and names, a flag specifying if store is active, and an array of image/logo sizes (relative URLs)"
    
    """
    url = f"https://cheapshark-game-deals.p.rapidapi.com/stores"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cheapshark-game-deals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manage_alerts(email: str, action: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send an email containing a link to manage your alerts."
    email: Any valid email address
        action: The action to take on the price alert, set to `manage`
        
    """
    url = f"https://cheapshark-game-deals.p.rapidapi.com/alerts"
    querystring = {'email': email, 'action': action, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cheapshark-game-deals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_games(title: str='batman', exact: int=0, limit: int=60, steamappid: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of games that contain a given title or matches steamAppID. Response includes the cheapest current deal for each game."
    title: Search for a game by title
        exact: Default `0`

Flag to allow only exact string match for `title` parameter
        limit: Default `60`

The maximum number of games to return, up to `60`
        steamappid: Search for a game by Steam’s AppID - e.g. http://store.steampowered.com/app/35140/
        
    """
    url = f"https://cheapshark-game-deals.p.rapidapi.com/games"
    querystring = {}
    if title:
        querystring['title'] = title
    if exact:
        querystring['exact'] = exact
    if limit:
        querystring['limit'] = limit
    if steamappid:
        querystring['steamAppID'] = steamappid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cheapshark-game-deals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_lookup(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets info for a specific game. Response includes a list of all deals associated with the game."
    id: An existing gameID
        
    """
    url = f"https://cheapshark-game-deals.p.rapidapi.com/games"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cheapshark-game-deals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def edit_alert(action: str, email: str, gameid: int, price: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Set or remove a price alert."
    action: The action to take on the price alert (`set` or `delete`)
        email: Any valid email address
        gameid: An existing gameID
        price: The price to wait for, only required when using `set` value for `action` parameter
        
    """
    url = f"https://cheapshark-game-deals.p.rapidapi.com/alerts"
    querystring = {'action': action, 'email': email, 'gameID': gameid, 'price': price, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cheapshark-game-deals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deal_lookup(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get info for a specific deal. Response includes game info, any cheaper current deals, and the cheapest historical price. As elsewhere, dealID is encoded"
    id: An Encoded Deal ID
        
    """
    url = f"https://cheapshark-game-deals.p.rapidapi.com/deals"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cheapshark-game-deals.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

