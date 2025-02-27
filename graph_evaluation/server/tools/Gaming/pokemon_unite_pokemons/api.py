import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_all_pokemons(pagesize: str='10', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all pokemons within pokemon unite"
    
    """
    url = f"https://pokemon-unite-pokemons.p.rapidapi.com/pokemon"
    querystring = {}
    if pagesize:
        querystring['pageSize'] = pagesize
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-unite-pokemons.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pokemon_data_by_name(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Pokemon data sing the pokemon name"
    
    """
    url = f"https://pokemon-unite-pokemons.p.rapidapi.com/pokemon/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pokemon-unite-pokemons.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

