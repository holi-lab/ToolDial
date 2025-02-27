import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gets_the_meta_data_from_one_given_champion_based_on_a_rank(rankname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetches the Data from the provided League of Legends Champion based on a rank.
		Will return a list of Objects, because a Champion can be played in multiple roles
		
		Possible rank parameters are: 
		- placements
		- iron
		- bronze
		- silver
		- gold
		- platinum
		- diamond
		- master
		- grandmaster
		- challenger
		They also be combined with a "comma-sign": e.g challenger,master"
    
    """
    url = f"https://league-of-legends-champion-meta.p.rapidapi.com/champions/{name}/rank/{rankname}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-champion-meta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gets_all_the_meta_data_from_all_champions_based_on_a_rank(rankname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetches all the Meta Data in a provided rank.
		Possible parameters are:
		- placements
		- iron
		- bronze
		- silver
		- gold
		- platinum
		- diamond
		- master
		- grandmaster
		- challenger
		They also can be combined with a 'comma-sign': e.g. challenger**,**master"
    
    """
    url = f"https://league-of-legends-champion-meta.p.rapidapi.com/champions/all/rank/{rankname}"
    querystring = {}
    if rankname:
        querystring['rankname'] = rankname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-champion-meta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gets_the_meta_data_from_one_given_champion(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetches the Data from the provided League of Legends Champion.
		Will return a list of Objects, because a Champion can be played in multiple roles"
    
    """
    url = f"https://league-of-legends-champion-meta.p.rapidapi.com/champions/{name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-champion-meta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gets_the_data_from_all_league_champions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetches the Meta Data for all available League of Legends Champions"
    
    """
    url = f"https://league-of-legends-champion-meta.p.rapidapi.com/champions/all"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-champion-meta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gets_all_the_available_champions_names_without_any_meta_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetches all the champion names"
    
    """
    url = f"https://league-of-legends-champion-meta.p.rapidapi.com/champions/all/names"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "league-of-legends-champion-meta.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

