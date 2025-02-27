import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fixture_tree_inplay_sport_sport_id(sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a tree list of fixtures suitable for constructing a SportBook navigation side menu.
		Response tree consists of sport, region, competition and fixtures.
		"
    
    """
    url = f"https://betbro-sportbook.p.rapidapi.com/fixture_tree/InPlay/sport/{sport_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betbro-sportbook.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixture_tree_pregame_sport_sport_id(sport_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a tree list of fixtures suitable for constructing a SportBook navigation side menu.
		Response tree consists of sport, region, competition and fixtures.
		"
    
    """
    url = f"https://betbro-sportbook.p.rapidapi.com/fixture_tree/PreGame/sport/{sport_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betbro-sportbook.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixture_fixture_id(fixture_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full fixture SportBook
		what the fuck is going on !
		"
    
    """
    url = f"https://betbro-sportbook.p.rapidapi.com/fixture/{fixture_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "betbro-sportbook.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

