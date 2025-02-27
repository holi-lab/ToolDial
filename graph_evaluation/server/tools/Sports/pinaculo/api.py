import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def league_matchups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league matchups."
    id: The league id.
        
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/leagues/{is_id}/matchups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_leagues(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search leagues."
    query: Query string.
        
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/leagues/search"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_matchups(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search matchups."
    query: Query string.
        
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/matchups/search"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sport_live_matchups(is_id: int, withspecials: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sport live matchups."
    id: The sport id.
        withspecials: Return with specials?
        
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/sports/{is_id}/matchups/live"
    querystring = {}
    if withspecials:
        querystring['withSpecials'] = withspecials
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def enums(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get enums."
    
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/enums"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sport_markets_straight(is_id: int, withspecials: bool=None, primaryonly: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sport market straight."
    id: The sport id.
        withspecials: Return only with specials?
        primaryonly: Return only primary?
        
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/sports/{is_id}/markets/straight"
    querystring = {}
    if withspecials:
        querystring['withSpecials'] = withspecials
    if primaryonly:
        querystring['primaryOnly'] = primaryonly
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sport_highlighted_markets_straight(is_id: int, onlyprimary: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sport highlighted market straight"
    id: The sport id.
        onlyprimary: Return primary only?
        
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/sports/{is_id}/markets/highlighted/straight"
    querystring = {}
    if onlyprimary:
        querystring['onlyPrimary'] = onlyprimary
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sport_translations(is_id: int, lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sport translations."
    id: The sport id.
        lang: The language.
        
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/sports/{is_id}/language/{lang}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchup_related_markets_straight(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get matchup related markets straight."
    id: The matchup id. Please use returned ids from other endpoints.
        
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/matchups/{is_id}/markets/related/straight"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sport_markets_live_straight(is_id: int, primaryonly: bool=None, withspecials: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sport markets live straight."
    id: The sport id.
        primaryonly: Return primary only?
        withspecials: Return with specials?
        
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/sports/{is_id}/markets/live/straight"
    querystring = {}
    if primaryonly:
        querystring['primaryOnly'] = primaryonly
    if withspecials:
        querystring['withSpecials'] = withspecials
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def labels(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get labels."
    
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/labels"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def league_markets_straight(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get league markets straight."
    id: The league id.
        
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/leagues/{is_id}/markets/straight"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def matchup_related(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get matchup related."
    id: The matchup id. Please use the ids returned from other endpoints.
        
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/matchups/{is_id}/related"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_sports(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live sports."
    
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/sports/live"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sport_matchups(is_id: int, withspecials: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sport matchups."
    id: The sport id?
        withspecials: Return with specials?
        
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/sports/{is_id}/matchups"
    querystring = {}
    if withspecials:
        querystring['withSpecials'] = withspecials
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sport_leagues(is_id: int, all: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sport leagues."
    id: The sport id.
        all: Return all?
        
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/sports/{is_id}/leagues"
    querystring = {}
    if all:
        querystring['all'] = all
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sports(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sports."
    
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/sports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sport_highlighted_matchups(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sport highlighted matchups."
    id: The sport id.
        
    """
    url = f"https://pinaculo.p.rapidapi.com/api/pinaculo/sports/{is_id}/matchups/highlighted"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinaculo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

