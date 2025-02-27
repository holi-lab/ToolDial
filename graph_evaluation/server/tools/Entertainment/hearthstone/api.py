import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def info(callback: str=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of current patch, classes, sets, types, factions, qualities, races and locales."
    callback: Request data to be returned as a JsonP callback.
        locale: What locale to use in the response. Default locale is enUS. Available locales: enUS, enGB, deDE, esES, esMX, frFR, itIT, koKR, plPL, ptBR, ruRU, zhCN, zhTW, jaJP, thTH.
        
    """
    url = f"https://omgvamp-hearthstone-v1.p.rapidapi.com/info"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "omgvamp-hearthstone-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def single_card(name: str, callback: str=None, collectible: int=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns card by name or ID. This may return more then one card if they share the same name. Loatheb returns both the card and the boss."
    name: The name or ID of the card. Example values: Ysera, EX1_572.
        callback: Request data to be returned as a JsonP callback.
        collectible: Set this to 1 to only return collectible cards.
        locale: What locale to use in the response. Default locale is enUS. Available locales: enUS, enGB, deDE, esES, esMX, frFR, itIT, koKR, plPL, ptBR, ruRU, zhCN, zhTW, jaJP, thTH.
        
    """
    url = f"https://omgvamp-hearthstone-v1.p.rapidapi.com/cards/{name}"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    if collectible:
        querystring['collectible'] = collectible
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "omgvamp-hearthstone-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cards_by_class(is_class: str, health: int=None, durability: int=None, cost: int=None, attack: int=None, callback: str=None, collectible: int=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the cards of a class. Example values: Mage, Paladin."
    class: Card class.
        health: Return only cards with a certain health.
        durability: Return only cards with a certain durability.
        cost: Return only cards of a certain cost.
        attack: Return only cards with a certain attack.
        callback: Request data to be returned as a JsonP callback.
        collectible: Set this to 1 to only return collectible cards.
        locale: What locale to use in the response. Default locale is enUS. Available locales: enUS, enGB, deDE, esES, esMX, frFR, itIT, koKR, plPL, ptBR, ruRU, zhCN, zhTW, jaJP, thTH.
        
    """
    url = f"https://omgvamp-hearthstone-v1.p.rapidapi.com/cards/classes/{is_class}"
    querystring = {}
    if health:
        querystring['health'] = health
    if durability:
        querystring['durability'] = durability
    if cost:
        querystring['cost'] = cost
    if attack:
        querystring['attack'] = attack
    if callback:
        querystring['callback'] = callback
    if collectible:
        querystring['collectible'] = collectible
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "omgvamp-hearthstone-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cards_by_race(race: str, attack: int=None, durability: int=None, health: int=None, cost: int=None, callback: str=None, collectible: int=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the cards of a certain race. Example values: Mech, Murloc."
    race: Card race.
        attack: Return only cards with a certain attack.
        durability: Return only cards with a certain durability.
        health: Return only cards with a certain health.
        cost: Return only cards of a certain cost.
        callback: Request data to be returned as a JsonP callback.
        collectible: Set this to 1 to only return collectible cards.
        locale: What locale to use in the response. Default locale is enUS. Available locales: enUS, enGB, deDE, esES, esMX, frFR, itIT, koKR, plPL, ptBR, ruRU, zhCN, zhTW, jaJP, thTH.
        
    """
    url = f"https://omgvamp-hearthstone-v1.p.rapidapi.com/cards/races/{race}"
    querystring = {}
    if attack:
        querystring['attack'] = attack
    if durability:
        querystring['durability'] = durability
    if health:
        querystring['health'] = health
    if cost:
        querystring['cost'] = cost
    if callback:
        querystring['callback'] = callback
    if collectible:
        querystring['collectible'] = collectible
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "omgvamp-hearthstone-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def card_set(set: str, cost: int=None, attack: int=None, durability: int=None, health: int=None, callback: str=None, collectible: int=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all cards in a set. Example values: Blackrock Mountain, Classic."
    set: Card set.
        cost: Return only cards of a certain cost.
        attack: Return only cards with a certain attack.
        durability: Return only cards with a certain durability.
        health: Return only cards with a certain health.
        callback: Request data to be returned as a JsonP callback.
        collectible: Set this to 1 to only return collectible cards.
        locale: What locale to use in the response. Default locale is enUS. Available locales: enUS, enGB, deDE, esES, esMX, frFR, itIT, koKR, plPL, ptBR, ruRU, zhCN, zhTW, jaJP, thTH.
        
    """
    url = f"https://omgvamp-hearthstone-v1.p.rapidapi.com/cards/sets/{set}"
    querystring = {}
    if cost:
        querystring['cost'] = cost
    if attack:
        querystring['attack'] = attack
    if durability:
        querystring['durability'] = durability
    if health:
        querystring['health'] = health
    if callback:
        querystring['callback'] = callback
    if collectible:
        querystring['collectible'] = collectible
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "omgvamp-hearthstone-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cards_by_quality(quality: str, cost: int=None, attack: int=None, durability: int=None, health: int=None, callback: str=None, collectible: int=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the cards of a certain quality. Example values: Legendary, Common."
    quality: Card quality.
        cost: Return only cards of a certain cost.
        attack: Return only cards with a certain attack.
        durability: Return only cards with a certain durability.
        health: Return only cards with a certain health.
        callback: Request data to be returned as a JsonP callback.
        collectible: Set this to 1 to only return collectible cards.
        locale: What locale to use in the response. Default locale is enUS. Available locales: enUS, enGB, deDE, esES, esMX, frFR, itIT, koKR, plPL, ptBR, ruRU, zhCN, zhTW, jaJP, thTH.
        
    """
    url = f"https://omgvamp-hearthstone-v1.p.rapidapi.com/cards/qualities/{quality}"
    querystring = {}
    if cost:
        querystring['cost'] = cost
    if attack:
        querystring['attack'] = attack
    if durability:
        querystring['durability'] = durability
    if health:
        querystring['health'] = health
    if callback:
        querystring['callback'] = callback
    if collectible:
        querystring['collectible'] = collectible
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "omgvamp-hearthstone-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def card_backs(callback: str=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all the card backs."
    callback: Request data to be returned as a JsonP callback.
        locale: What locale to use in the response. Default locale is enUS. Available locales: enUS, enGB, deDE, esES, esMX, frFR, itIT, koKR, plPL, ptBR, ruRU, zhCN, zhTW, jaJP, thTH.
        
    """
    url = f"https://omgvamp-hearthstone-v1.p.rapidapi.com/cardbacks"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "omgvamp-hearthstone-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def card_search(name: str, callback: str=None, collectible: int=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns cards by partial name."
    callback: Request data to be returned as a JsonP callback.
        collectible: Set this to 1 to only return collectible cards.
        locale: What locale to use in the response. Default locale is enUS. Available locales: enUS, enGB, deDE, esES, esMX, frFR, itIT, koKR, plPL, ptBR, ruRU, zhCN, zhTW, jaJP, thTH.
        
    """
    url = f"https://omgvamp-hearthstone-v1.p.rapidapi.com/cards/search/{name}"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    if collectible:
        querystring['collectible'] = collectible
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "omgvamp-hearthstone-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cards_by_faction(faction: str, attack: int=None, durability: int=None, health: int=None, callback: str=None, collectible: int=None, cost: int=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the cards of a certain faction. Example values: Horde, Neutral."
    faction: Card faction.
        attack: Return only cards with a certain attack.
        durability: Return only cards with a certain durability.
        health: Return only cards with a certain health.
        callback: Request data to be returned as a JsonP callback.
        collectible: Set this to 1 to only return collectible cards.
        cost: Return only cards of a certain cost.
        locale: What locale to use in the response. Default locale is enUS. Available locales: enUS, enGB, deDE, esES, esMX, frFR, itIT, koKR, plPL, ptBR, ruRU, zhCN, zhTW, jaJP, thTH.
        
    """
    url = f"https://omgvamp-hearthstone-v1.p.rapidapi.com/cards/factions/{faction}"
    querystring = {}
    if attack:
        querystring['attack'] = attack
    if durability:
        querystring['durability'] = durability
    if health:
        querystring['health'] = health
    if callback:
        querystring['callback'] = callback
    if collectible:
        querystring['collectible'] = collectible
    if cost:
        querystring['cost'] = cost
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "omgvamp-hearthstone-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cards_by_type(type: str, cost: int=None, attack: int=None, durability: int=None, health: int=None, callback: str=None, collectible: int=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the cards of a certain type. Example values: Spell, Weapon."
    type: Card type.
        cost: Return only cards of a certain cost.
        attack: Return only cards with a certain attack.
        durability: Return only cards with a certain durability.
        health: Return only cards with a certain health.
        callback: Request data to be returned as a JsonP callback.
        collectible: Set this to 1 to only return collectible cards.
        locale: What locale to use in the response. Default locale is enUS. Available locales: enUS, enGB, deDE, esES, esMX, frFR, itIT, koKR, plPL, ptBR, ruRU, zhCN, zhTW, jaJP, thTH.
        
    """
    url = f"https://omgvamp-hearthstone-v1.p.rapidapi.com/cards/types/{type}"
    querystring = {}
    if cost:
        querystring['cost'] = cost
    if attack:
        querystring['attack'] = attack
    if durability:
        querystring['durability'] = durability
    if health:
        querystring['health'] = health
    if callback:
        querystring['callback'] = callback
    if collectible:
        querystring['collectible'] = collectible
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "omgvamp-hearthstone-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_cards(cost: int=None, attack: int=None, health: int=None, callback: str=None, collectible: int=None, durability: int=None, locale: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all available Hearthstone cards including non collectible cards."
    cost: Return only cards of a certain cost.
        attack: Return only cards with a certain attack.
        health: Return only cards with a certain health.
        callback: Request data to be returned as a JsonP callback.
        collectible: Set this to 1 to only return collectible cards.
        durability: Return only cards with a certain durability.
        locale: What locale to use in the response. Default locale is enUS. Available locales: enUS, enGB, deDE, esES, esMX, frFR, itIT, koKR, plPL, ptBR, ruRU, zhCN, zhTW, jaJP, thTH.
        
    """
    url = f"https://omgvamp-hearthstone-v1.p.rapidapi.com/cards"
    querystring = {}
    if cost:
        querystring['cost'] = cost
    if attack:
        querystring['attack'] = attack
    if health:
        querystring['health'] = health
    if callback:
        querystring['callback'] = callback
    if collectible:
        querystring['collectible'] = collectible
    if durability:
        querystring['durability'] = durability
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "omgvamp-hearthstone-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

