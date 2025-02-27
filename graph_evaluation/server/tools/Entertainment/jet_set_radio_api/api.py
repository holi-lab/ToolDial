import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def graffiti_tags(orderby: str='asc', limit: str='5', sortby: str='name', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all Graffiti-Tags"
    orderby: Possible Values:
 - `asc`
 - `desc`

sortBy must be present for this to take effect. 
The default value is `asc`

        limit: Limit the number of items returned in the response.
        sortby: Possible Values:
Any field on the document

Examples:
 - name
 - stats
        
    """
    url = f"https://jet-set-radio-api.p.rapidapi.com/graffiti-tags"
    querystring = {}
    if orderby:
        querystring['orderBy'] = orderby
    if limit:
        querystring['limit'] = limit
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jet-set-radio-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def artists(limit: str='5', sortby: str='name', orderby: str='asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all music Artists who contributed to JSR/JSRF"
    limit: Limit the number of items returned in the response.
        sortby: Possible Values:
Any field on the document

Examples:
 - name
 - stats
        orderby: Possible Values:
 - `asc`
 - `desc`

sortBy must be present for this to take effect. 
The default value is `asc`

        
    """
    url = f"https://jet-set-radio-api.p.rapidapi.com/artists"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if sortby:
        querystring['sortBy'] = sortby
    if orderby:
        querystring['orderBy'] = orderby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jet-set-radio-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songs(limit: str='5', sortby: str='name', orderby: str='asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all Songs from JSR/JSRF"
    limit: Limit the number of items returned in the response.
        sortby: Possible Values:
Any field on the document

Examples:
 - name
 - stats
        orderby: Possible Values:
 - `asc`
 - `desc`

sortBy must be present for this to take effect. 
The default value is `asc`

        
    """
    url = f"https://jet-set-radio-api.p.rapidapi.com/songs"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if sortby:
        querystring['sortBy'] = sortby
    if orderby:
        querystring['orderBy'] = orderby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jet-set-radio-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def levels(sortby: str='name', limit: str='5', orderby: str='asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all Levels"
    sortby: Possible Values:
Any field on the document

Examples:
 - name
 - stats
        limit: Limit the number of items returned in the response.
        orderby: Possible Values:
 - `asc`
 - `desc`

sortBy must be present for this to take effect. 
The default value is `asc`

        
    """
    url = f"https://jet-set-radio-api.p.rapidapi.com/levels"
    querystring = {}
    if sortby:
        querystring['sortBy'] = sortby
    if limit:
        querystring['limit'] = limit
    if orderby:
        querystring['orderBy'] = orderby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jet-set-radio-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations(sortby: str='name', limit: str='5', orderby: str='asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all Locations in JSR/JSRF"
    sortby: Possible Values:
Any field on the document

Examples:
 - name
 - stats
        limit: Limit the number of items returned in the response.
        orderby: Possible Values:
 - `asc`
 - `desc`

sortBy must be present for this to take effect. 
The default value is `asc`

        
    """
    url = f"https://jet-set-radio-api.p.rapidapi.com/locations"
    querystring = {}
    if sortby:
        querystring['sortBy'] = sortby
    if limit:
        querystring['limit'] = limit
    if orderby:
        querystring['orderBy'] = orderby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jet-set-radio-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def games(limit: str='5', orderby: str='asc', sortby: str='name', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all Games"
    limit: Limit the number of items returned in the response.
        orderby: Possible Values:
 - `asc`
 - `desc`

sortBy must be present for this to take effect. 
The default value is `asc`

        sortby: Possible Values:
Any field on the document

Examples:
 - name
 - stats
        
    """
    url = f"https://jet-set-radio-api.p.rapidapi.com/games"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if orderby:
        querystring['orderBy'] = orderby
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jet-set-radio-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jsrf_character_by_id(characterid: str, limit: str='5', orderby: str='asc', sortby: str='name', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single Character by ID"
    limit: Limit the number of items returned in the response.
        orderby: Possible Values:
 - `asc`
 - `desc`

sortBy must be present for this to take effect. 
The default value is `asc`

        sortby: Possible Values:
Any field on the document

Examples:
 - name
 - stats
        
    """
    url = f"https://jet-set-radio-api.p.rapidapi.com/characters/jsrf/{characterid}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if orderby:
        querystring['orderBy'] = orderby
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jet-set-radio-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jsrf_characters(limit: str='5', orderby: str='asc', sortby: str='name', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all JSRF Characters"
    limit: Limit the number of items returned in the response.
        orderby: Possible Values:
 - `asc`
 - `desc`

sortBy must be present for this to take effect. 
The default value is `asc`

        sortby: Possible Values:
Any field on the document

Examples:
 - name
 - stats
        
    """
    url = f"https://jet-set-radio-api.p.rapidapi.com/characters/jsrf"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if orderby:
        querystring['orderBy'] = orderby
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jet-set-radio-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jsr_character_by_id(characterid: str, orderby: str='asc', limit: str='5', sortby: str='name', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single Character by ID"
    orderby: Possible Values:
 - `asc`
 - `desc`

sortBy must be present for this to take effect. 
The default value is `asc`

        limit: Limit the number of items returned in the response.
        sortby: Possible Values:
Any field on the document

Examples:
 - name
 - stats
        
    """
    url = f"https://jet-set-radio-api.p.rapidapi.com/characters/jsr/{characterid}"
    querystring = {}
    if orderby:
        querystring['orderBy'] = orderby
    if limit:
        querystring['limit'] = limit
    if sortby:
        querystring['sortBy'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jet-set-radio-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def characters(limit: str='5', sortby: str='name', orderby: str='asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all Characters"
    limit: Limit the number of items returned in the response.
        sortby: Possible Values:
Any field on the document

Examples:
 - name
 - stats
        orderby: Possible Values:
 - `asc`
 - `desc`

sortBy must be present for this to take effect. 
The default value is `asc`

        
    """
    url = f"https://jet-set-radio-api.p.rapidapi.com/characters"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if sortby:
        querystring['sortBy'] = sortby
    if orderby:
        querystring['orderBy'] = orderby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jet-set-radio-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jsr_characters(limit: str='5', sortby: str='name', orderby: str='asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all JSR Characters"
    limit: Limit the number of items returned in the response.
        sortby: Possible Values:
Any field on the document

Examples:
 - name
 - stats
        orderby: Possible Values:
 - `asc`
 - `desc`

sortBy must be present for this to take effect. 
The default value is `asc`

        
    """
    url = f"https://jet-set-radio-api.p.rapidapi.com/characters/jsr"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if sortby:
        querystring['sortBy'] = sortby
    if orderby:
        querystring['orderBy'] = orderby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jet-set-radio-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

