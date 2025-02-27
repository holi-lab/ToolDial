import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def game_data_origin_patches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get IDs of different patches of origin game version"
    
    """
    url = f"https://axie-data-hub.p.rapidapi.com/game-data/origin/patches"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_data_classic_effects_effecttype(effecttype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all ability card effects of a specific type (buffs, debuffs)."
    effecttype: Effect type (buffs, debuffs).
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/game-data/classic/effects/{effecttype}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange_symbol(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get exchange data of a specific asset symbol."
    symbol: Asset symbol.
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/exchange/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_data_origin_cards(format: str='default', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get origin ability cards data. You can filter by any valid card object property, such as 'class', 'type' or 'abilityType'. E.g. '/cards/origin?class=reptile&type=mouth&abilityType=AttackRanged'. Theese filters are not available if you use 'format=original'"
    format: Format of the response: 'default' or 'original'. Original has the original format response given by Axie Infinity. Default is a better formatted version of the original response. Default format is 'default'.
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/game-data/origin/cards"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_data_classic_cards_patches(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get info of different cards balance patches of classic game version."
    
    """
    url = f"https://axie-data-hub.p.rapidapi.com/game-data/classic/cards/patches"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_data_origin_cards_patchid(patchid: str, format: str='default', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get origin ability cards data from specific patch. You can filter by any valid card object property, such as 'class', 'type' or 'abilityType'. E.g. '/cards/origin?class=reptile&type=mouth&abilityType=AttackRanged'. Theese filters are not available if you use 'format=original'"
    patchid: Patch ID.
        format: Format of the response: 'default' or 'original'. Original has the original format response given by Axie Infinity. Default is a better formatted version of the original response. Default format is 'default'.
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/game-data/origin/cards/{patchid}"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_data_origin_tools_patchid(patchid: str, format: str='default', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get origin card tools data from specific patch."
    patchid: Patch ID.
        format: Format of the response: 'default' or 'original'. Original has the original format response given by Axie Infinity. Default is a better formatted version of the original response. Default format is 'default'.
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/game-data/origin/tools/{patchid}"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_data_classic_effects(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all ability card effects."
    
    """
    url = f"https://axie-data-hub.p.rapidapi.com/game-data/classic/effects"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_data_classic_cards_patchid(patchid: str, format: str='default', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get classic ability cards data from specific patch. You can filter by any valid card object property, such as 'class' or 'type'. E.g. '/cards/classic/current?class=reptile&type=mouth'. Theese filters are not available if you use 'format=original'"
    patchid: Patch ID.
        format: Format of the response: 'default' or 'original'. Original has the original format response given by Axie Infinity. Default is a better formatted version of the original response. Default format is 'default'.
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/game-data/classic/cards/{patchid}"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def version(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current API data source version. This version may be different as the version of Axie Data Hub API deployed on RapidAPI."
    
    """
    url = f"https://axie-data-hub.p.rapidapi.com/version"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def axie_axieid_children(axieid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an axie's children data."
    axieid: Axie ID
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/axie/{axieid}/children"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_data_origin_tools(format: str='default', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get origin card tools data."
    format: Format of the response: 'default' or 'original'. Original has the original format response given by Axie Infinity. Default is a better formatted version of the original response. Default format is 'default'.
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/game-data/origin/tools"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_data_classic_stats_body_part_classname(classname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stats added by body parts of specific class."
    classname: Axie class name.
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/game-data/classic/stats/body-part/{classname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_data_classic_stats_body_part(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stats added by body parts of each class."
    
    """
    url = f"https://axie-data-hub.p.rapidapi.com/game-data/classic/stats/body-part"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_data_classic_stats_base_classname(classname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get base stats of a specific axie class."
    classname: Axie class name.
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/game-data/classic/stats/base/{classname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def game_data_classic_stats_base(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get axie classes base stats."
    
    """
    url = f"https://axie-data-hub.p.rapidapi.com/game-data/classic/stats/base"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def axie_axieid(axieid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an axie details given his ID."
    axieid: Axie ID
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/axie/{axieid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auction_sold(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data of latest sold axies."
    
    """
    url = f"https://axie-data-hub.p.rapidapi.com/auction/sold"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def axie_axieid_parts(axieid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an axie's parts data."
    axieid: Axie ID
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/axie/{axieid}/parts"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def axie_axieid_name(axieid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an axie's name."
    axieid: Axie ID
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/axie/{axieid}/name"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def axie_axieid_genes(axieid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an axie's genes."
    axieid: Axie ID
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/axie/{axieid}/genes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaderboard(is_from: int=1, to: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get MMR leaderboard data."
    is_from: Start rank index of the leaderboard.
        to: End rank index of the leaderboard.
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/leaderboard"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def axie_axieid_stats(axieid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an axie's stats data."
    axieid: Axie ID
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/axie/{axieid}/stats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaderboard_history(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of available season top 1000 cached leaderboards. This just returns the available season IDs."
    
    """
    url = f"https://axie-data-hub.p.rapidapi.com/leaderboard/history"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_address_battles(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a player's battles data."
    address: Player's ronin address
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/player/{address}/battles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_address_wallet_transactions(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a player's transactions data."
    address: Player's ronin address
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/player/{address}/wallet/transactions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaderboard_previous(to: int=100, is_from: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get MMR leaderboard data of the previous season."
    to: End rank index of the leaderboard.
        is_from: Start rank index of the leaderboard.
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/leaderboard/previous"
    querystring = {}
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_address_axies(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a player's axies data."
    address: Player's ronin address
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/player/{address}/axies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auction_onsale(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data of latest axies on sale."
    
    """
    url = f"https://axie-data-hub.p.rapidapi.com/auction/onsale"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_address_battles_type(type: str, address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a player's battles data of a given type (pvp or pve)."
    type: Battle type (pvp or pve)
        address: Player's ronin address
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/player/{address}/battles/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_address_mmr(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a player's mmr data."
    address: Player's ronin address
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/player/{address}/mmr"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_address_name(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a player's name."
    address: Player's ronin address
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/player/{address}/name"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_address_mmr_previous(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a player's mmr of previous season."
    address: Player's ronin address
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/player/{address}/mmr/previous"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def leaderboard_history_seasonid(seasonid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top 1000 MMR leaderboard data of a specific season."
    seasonid: Season ID.
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/leaderboard/history/{seasonid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_address(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a player's data given his ronin address."
    address: Player's ronin address
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/player/{address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def player_address_wallet_tokens(address: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a player's tokens data."
    address: Player's ronin address
        
    """
    url = f"https://axie-data-hub.p.rapidapi.com/player/{address}/wallet/tokens"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axie-data-hub.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

