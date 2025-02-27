import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def pagination(op: int, page: int, idol: str='Rachel Love,Kenna James', source: str='jjgirls.com,www.jkforum.net', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get's Pagination data from db"
    op: Number of items per page
        page: Page
        idol: idol Filter,      use comma to use multiple idol as filter
        source: Source filter,      use comma to use multiple source as filter
        
    """
    url = f"https://bloom-hina.p.rapidapi.com/hina"
    querystring = {'op': op, 'page': page, }
    if idol:
        querystring['idol'] = idol
    if source:
        querystring['source'] = source
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloom-hina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, page: int, idol: str=None, source: str=None, op: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets Text query data from db
		
		Strict Termed Searches are to be made with a double quote 
		     eg: "Skinny"
		Exclusion are to be made in this format after any prefilter
		     eg: "thick" ~"skinny""
    query: search term
        page: Page number
        idol: idol Filter,      use comma to use multiple idol as filter
        source: Source filter,      use comma to use multiple source as filter
        op: Number of items per page
        
    """
    url = f"https://bloom-hina.p.rapidapi.com/hina/search"
    querystring = {'query': query, 'page': page, }
    if idol:
        querystring['idol'] = idol
    if source:
        querystring['source'] = source
    if op:
        querystring['op'] = op
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloom-hina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def idols(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://bloom-hina.p.rapidapi.com/hina/idols"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloom-hina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pollen(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "4X A.I Images, endpoint is accounted for usage."
    id: ID of the album
        
    """
    url = f"https://bloom-hina.p.rapidapi.com/hina/pollen/payload"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloom-hina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random(many: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets Random Album/s"
    many: gets many albums
        
    """
    url = f"https://bloom-hina.p.rapidapi.com/hina/random"
    querystring = {'many': many, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloom-hina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def album(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets Album from db"
    id: Id of the album
        
    """
    url = f"https://bloom-hina.p.rapidapi.com/hina/payload"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloom-hina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sources(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a List of sources Hina Supports"
    
    """
    url = f"https://bloom-hina.p.rapidapi.com/hina/sources"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloom-hina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resize(num: int, is_id: str, width: int, height: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets Resized image of album from db"
    num: Which Image to resize
        id: Path param id of the album
        width: Output Width (Optional)
        height: Output Height (Optional)
        
    """
    url = f"https://bloom-hina.p.rapidapi.com/strat/thumbs/{is_id}/{num}"
    querystring = {'width': width, 'height': height, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloom-hina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def thumb(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets Thumb of the album"
    
    """
    url = f"https://bloom-hina.p.rapidapi.com/hina/{is_id}/thumb"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bloom-hina.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

