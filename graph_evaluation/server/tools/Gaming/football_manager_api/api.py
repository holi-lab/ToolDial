import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_players_of_club(clubname: str, length: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get players that play for a club. Club name is the "Short Name" used in game without spaces. Examples: "Barcelona", "ManCity", "Liverpool", "PSG""
    
    """
    url = f"https://football-manager-api.p.rapidapi.com/clubs/{clubname}"
    querystring = {}
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-manager-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_similarity_of_players(firstplayer: str, secondplayer: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the similarity of the two players supplied."
    
    """
    url = f"https://football-manager-api.p.rapidapi.com/players/similar/{firstplayer}/{secondplayer}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-manager-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_weighted_cosine_similar_player(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get most similar based on weighted cosine similarity."
    
    """
    url = f"https://football-manager-api.p.rapidapi.com/players/similar/cosine/alternative/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-manager-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cosine_similar_player(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get most similar player by cosine similarity."
    
    """
    url = f"https://football-manager-api.p.rapidapi.com/players/similar/cosine/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-manager-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_similar_player(comparisonplayer: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Finds the player that has the most similar attributes to another player submitted in the query."
    
    """
    url = f"https://football-manager-api.p.rapidapi.com/players/similar/{comparisonplayer}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-manager-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def players(orderby: str='currentAbility-desc', minaskingprice: int=10, minage: int=16, minca: int=130, maxaskingprice: int=200, name: str='Messi', nationality: str='ARG', position: str='ST (C)', length: int=20, club: str='PSG', contractto: int=2025, maxage: int=40, minpa: int=170, maxca: int=200, maxpa: int=200, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Filter through the Football Manager players by their age, asking Price, club, minimum CA (Current Ability, 1-200), maximum CA, minimum PA (Potential Ability, 1-200), maximum PA, name, End of Contract year, nationality, position (and secondary positions), and select the length of entries you want to receive. Specify an attribute to sort the results by in this format: "finishing-desc", "InjuryProneness-asc"
		
		See Documentation for the filter query attributes dictionary."
    
    """
    url = f"https://football-manager-api.p.rapidapi.com/players"
    querystring = {}
    if orderby:
        querystring['orderBy'] = orderby
    if minaskingprice:
        querystring['minAskingPrice'] = minaskingprice
    if minage:
        querystring['minAge'] = minage
    if minca:
        querystring['minCa'] = minca
    if maxaskingprice:
        querystring['maxAskingPrice'] = maxaskingprice
    if name:
        querystring['name'] = name
    if nationality:
        querystring['nationality'] = nationality
    if position:
        querystring['position'] = position
    if length:
        querystring['length'] = length
    if club:
        querystring['club'] = club
    if contractto:
        querystring['contractTo'] = contractto
    if maxage:
        querystring['maxAge'] = maxage
    if minpa:
        querystring['minPa'] = minpa
    if maxca:
        querystring['maxCa'] = maxca
    if maxpa:
        querystring['maxPa'] = maxpa
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-manager-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_player_by_id(is_id: str, length: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a player by his unique ID."
    
    """
    url = f"https://football-manager-api.p.rapidapi.com/players/{is_id}"
    querystring = {}
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "football-manager-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

