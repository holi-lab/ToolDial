import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_champions(accept_encoding: str, season: str='2021', country: str='fr', constructor: str='yamaha', is_class: str='motogp', rider: str='fabio', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find Champions by season, rider, country, constructor and/or class."
    
    """
    url = f"https://motogp-data-api.p.rapidapi.com/champions/search"
    querystring = {'Accept-Encoding': accept_encoding, }
    if season:
        querystring['season'] = season
    if country:
        querystring['country'] = country
    if constructor:
        querystring['constructor'] = constructor
    if is_class:
        querystring['class'] = is_class
    if rider:
        querystring['rider'] = rider
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_one_champion(accept_encoding: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a specific Champion by ID."
    
    """
    url = f"https://motogp-data-api.p.rapidapi.com/champions/{is_id}"
    querystring = {'Accept-Encoding': accept_encoding, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_champions(accept_encoding: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets hundreds of results about the riders who have become world champions."
    
    """
    url = f"https://motogp-data-api.p.rapidapi.com/champions"
    querystring = {'Accept-Encoding': accept_encoding, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_poles(accept_encoding: str, rider: str='valentino', country: str='it', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find Poles by rider and/or country."
    
    """
    url = f"https://motogp-data-api.p.rapidapi.com/podiums/search"
    querystring = {'Accept-Encoding': accept_encoding, }
    if rider:
        querystring['rider'] = rider
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_one_pole(accept_encoding: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a specific Pole by ID."
    
    """
    url = f"https://motogp-data-api.p.rapidapi.com/poles/{is_id}"
    querystring = {'Accept-Encoding': accept_encoding, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_poles(accept_encoding: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets hundreds of results about the total number of poles a rider has completed."
    
    """
    url = f"https://motogp-data-api.p.rapidapi.com/poles"
    querystring = {'Accept-Encoding': accept_encoding, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_podiums(accept_encoding: str, rider: str='valentino', country: str='it', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find Podiums by rider and/or country."
    
    """
    url = f"https://motogp-data-api.p.rapidapi.com/podiums/search"
    querystring = {'Accept-Encoding': accept_encoding, }
    if rider:
        querystring['rider'] = rider
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_one_podium(accept_encoding: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a specific Podium by ID."
    
    """
    url = f"https://motogp-data-api.p.rapidapi.com/podiums/{is_id}"
    querystring = {'Accept-Encoding': accept_encoding, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_podiums(accept_encoding: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets hundreds of results about the total number of podiums a rider has completed."
    
    """
    url = f"https://motogp-data-api.p.rapidapi.com/podiums"
    querystring = {'Accept-Encoding': accept_encoding, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_fastest_laps(accept_encoding: str, rider: str='valentino', country: str='it', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find Fastest Laps by rider and/or country."
    
    """
    url = f"https://motogp-data-api.p.rapidapi.com/fastest-laps/search"
    querystring = {'Accept-Encoding': accept_encoding, }
    if rider:
        querystring['rider'] = rider
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_one_fastest_lap(accept_encoding: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a specific Fastest Lap by ID."
    
    """
    url = f"https://motogp-data-api.p.rapidapi.com/fastest-laps/{is_id}"
    querystring = {'Accept-Encoding': accept_encoding, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_fastest_laps(accept_encoding: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets hundreds of results about the total number of fastest laps a rider has completed."
    
    """
    url = f"https://motogp-data-api.p.rapidapi.com/fastest-laps"
    querystring = {'Accept-Encoding': accept_encoding, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_grand_prix_race_winners(accept_encoding: str, constructor: str='aprilia', season: str='1999', country: str='it', is_class: str='250', rider: str='valentino', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find Grand Prix race winners by rider, circuit, class, constructor, country or season."
    
    """
    url = f"https://motogp-data-api.p.rapidapi.com/winners/search"
    querystring = {'Accept-Encoding': accept_encoding, }
    if constructor:
        querystring['constructor'] = constructor
    if season:
        querystring['season'] = season
    if country:
        querystring['country'] = country
    if is_class:
        querystring['class'] = is_class
    if rider:
        querystring['rider'] = rider
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_one_grand_prix_race_winner(accept_encoding: str, is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a specific Grand Prix race winner by ID."
    
    """
    url = f"https://motogp-data-api.p.rapidapi.com/winners/{is_id}"
    querystring = {'Accept-Encoding': accept_encoding, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_grand_prix_race_winners(accept_encoding: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets hundreds of Grand Prix race winners."
    
    """
    url = f"https://motogp-data-api.p.rapidapi.com/winners"
    querystring = {'Accept-Encoding': accept_encoding, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "motogp-data-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

