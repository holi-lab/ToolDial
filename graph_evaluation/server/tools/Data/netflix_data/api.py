import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def season_episodes(ids: str, limit: str='25', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "season-episodes"
    ids: Season IDs (you can separate with commas)
        limit: Episode Limit
        offset: Offset
        
    """
    url = f"https://netflix-data.p.rapidapi.com/season/episodes/"
    querystring = {'ids': ids, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_trailers(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "title-trailers"
    id: Title ID
        
    """
    url = f"https://netflix-data.p.rapidapi.com/title/trailers/"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_similars(is_id: str, limit: str='25', offset: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "title-similars"
    id: Title ID
        limit: Title Limit
        offset: Offset
        
    """
    url = f"https://netflix-data.p.rapidapi.com/title/similars/"
    querystring = {'id': is_id, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_seasons(ids: str, limit: int=25, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "title-seasons"
    ids: Title IDs (you can separate with commas)
        limit: Season Limit
        offset: Offset
        
    """
    url = f"https://netflix-data.p.rapidapi.com/title/seasons/"
    querystring = {'ids': ids, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def title_details(ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "title-details"
    ids: Title IDs (you can separate with commas)
        
    """
    url = f"https://netflix-data.p.rapidapi.com/title/details/"
    querystring = {'ids': ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, offset: int=0, limit_suggestions: int=20, limit_titles: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search"
    query: Search Query
        offset: Offset
        limit_suggestions: Suggestion Limit
        limit_titles: Title Limit
        
    """
    url = f"https://netflix-data.p.rapidapi.com/search/"
    querystring = {'query': query, }
    if offset:
        querystring['offset'] = offset
    if limit_suggestions:
        querystring['limit_suggestions'] = limit_suggestions
    if limit_titles:
        querystring['limit_titles'] = limit_titles
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "netflix-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

