import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def season_episodes(ids: str, lang: str='en', limit: int=25, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Season Episodes"
    ids: Season IDs (you can separate with commas)
        limit: Episode Limit
        offset: Offset
        
    """
    url = f"https://netflix54.p.rapidapi.com/season/episodes/"
    querystring = {'ids': ids, }
    if lang:
        querystring['lang'] = lang
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix54.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_trailers(is_id: str, lang: str='en', limit: int=25, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Title Trailers"
    id: Title ID
        limit: Trailer Limit
        offset: Offset
        
    """
    url = f"https://netflix54.p.rapidapi.com/title/trailers/"
    querystring = {'id': is_id, }
    if lang:
        querystring['lang'] = lang
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix54.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_similars(is_id: str, limit: int=25, lang: str='en', offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Title Similars"
    id: Title ID
        limit: Title Limit
        offset: Offset
        
    """
    url = f"https://netflix54.p.rapidapi.com/title/similars/"
    querystring = {'id': is_id, }
    if limit:
        querystring['limit'] = limit
    if lang:
        querystring['lang'] = lang
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix54.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_seasons(ids: str, lang: str='en', offset: int=0, limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Title Seasons"
    ids: Title IDs (you can separate with commas)
        offset: Offset
        limit: Season Limit
        
    """
    url = f"https://netflix54.p.rapidapi.com/title/seasons/"
    querystring = {'ids': ids, }
    if lang:
        querystring['lang'] = lang
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix54.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_details(ids: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Title Details"
    ids: Title IDs (you can separate with commas)
        
    """
    url = f"https://netflix54.p.rapidapi.com/title/details/"
    querystring = {'ids': ids, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix54.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, lang: str='en', limit_titles: int=50, limit_suggestions: int=20, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search"
    query: Search Query
        limit_titles: Title Limit
        limit_suggestions: Suggestion Limit
        offset: Offset
        
    """
    url = f"https://netflix54.p.rapidapi.com/search/"
    querystring = {'query': query, }
    if lang:
        querystring['lang'] = lang
    if limit_titles:
        querystring['limit_titles'] = limit_titles
    if limit_suggestions:
        querystring['limit_suggestions'] = limit_suggestions
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix54.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def languages(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Languages"
    
    """
    url = f"https://netflix54.p.rapidapi.com/languages/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix54.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

