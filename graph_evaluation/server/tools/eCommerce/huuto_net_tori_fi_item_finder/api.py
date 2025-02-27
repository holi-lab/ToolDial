import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def huuto_net_and_tori_fi_query(searchterm: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Huuto.net and tori.fi query with chosen searchterm"
    
    """
    url = f"https://huuto-net-tori-fi-item-finder.p.rapidapi.com/kaikki/{searchterm}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "huuto-net-tori-fi-item-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def huuto_net_and_tori_fi_query_with_county(county: str, searchterm: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Huuto.net and tori.fi query with county"
    
    """
    url = f"https://huuto-net-tori-fi-item-finder.p.rapidapi.com/kaikki/{searchterm}/{county}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "huuto-net-tori-fi-item-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def huuto_net_query_with_county(searchterm: str, county: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Huuto.net query with county"
    
    """
    url = f"https://huuto-net-tori-fi-item-finder.p.rapidapi.com/huuto/{searchterm}/{county}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "huuto-net-tori-fi-item-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def huuto_net_query_with_a_search_term(searchterm: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Huuto.net query with a search term"
    
    """
    url = f"https://huuto-net-tori-fi-item-finder.p.rapidapi.com/huuto/{searchterm}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "huuto-net-tori-fi-item-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tori_fi_query_with_chosen_county(searchterm: str, county: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tori.fi query with chosen county"
    
    """
    url = f"https://huuto-net-tori-fi-item-finder.p.rapidapi.com/tori/{searchterm}/{county}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "huuto-net-tori-fi-item-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tori_fi_query(searchterm: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "gives a JSON of results with given searchterm"
    
    """
    url = f"https://huuto-net-tori-fi-item-finder.p.rapidapi.com/tori/{searchterm}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "huuto-net-tori-fi-item-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

