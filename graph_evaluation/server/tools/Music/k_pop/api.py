import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def girl_groups(q: str, by: str='Group Name', limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get info of girl-groups"
    q: Query to search
(Required)
        by: default: Stage Name
'by' parameter help you choose which info you would like to query based on.

Only the below values can be accepted for 'by'. 

- Group Name (default)
- Short Group Name
- Korean Name
- Date of Debut
- Company
- Members
- Original Members
- Fanclub Name
- Active
        limit: default: all there is
count limit for result
        offset: default: 0
offset for the result data
        
    """
    url = f"https://k-pop.p.rapidapi.com/girl-groups"
    querystring = {'q': q, }
    if by:
        querystring['by'] = by
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "k-pop.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_girl_group(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get random girl-group"
    
    """
    url = f"https://k-pop.p.rapidapi.com/girl-groups/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "k-pop.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_boy_group(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get random boy-group"
    
    """
    url = f"https://k-pop.p.rapidapi.com/boy-groups/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "k-pop.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def songs(q: str, by: str='Song Name', limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get K-POP songs info"
    q: Query to search
(Required)
        by: default: Song Name
'by' parameter help you choose which info you would like to query based on.

Only the below values can be accepted for \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"by\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\". 

- Song Name (default)
- Date
- Artist
- Korean Name
- Director
- Type
- Release

        limit: default: all there is
count limit for result
        offset: default: 0
offset for the result data
        
    """
    url = f"https://k-pop.p.rapidapi.com/songs"
    querystring = {'q': q, }
    if by:
        querystring['by'] = by
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "k-pop.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_idol(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get random idol"
    
    """
    url = f"https://k-pop.p.rapidapi.com/idols/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "k-pop.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def boy_groups(q: str, offset: int=None, by: str='Group Name', limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get boy-groups info"
    q: Query to search
(Required)
        offset: default: 0
offset for the result data
        by: default: Stage Name
'by' parameter help you choose which info you would like to query based on.

Only the below values can be accepted for 'by'. 

- Group Name (default)
- Short Group Name
- Korean Name
- Date of Debut
- Company
- Members
- Original Members
- Fanclub Name
- Active
        limit: default: all there is
count limit for result
        
    """
    url = f"https://k-pop.p.rapidapi.com/boy-groups"
    querystring = {'q': q, }
    if offset:
        querystring['offset'] = offset
    if by:
        querystring['by'] = by
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "k-pop.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random_song(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get random K-POP song"
    
    """
    url = f"https://k-pop.p.rapidapi.com/songs/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "k-pop.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def idols(q: str, by: str='Stage Name', limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get idol individual info"
    q: Query to search
(Required)
        by: default: Stage Name
'by' parameter help you choose which info you would like to query based on.

Only the below values can be accepted for 'by'

-  Stage Name (default)
- Full Name
- Korean Name
- K. Stage Name
- Date of Birth
- Group
- Country
- Second Country
- Height
- Weight
- Birthplace
- Other Group
- Former Group
- Gender
- Position
- Instagram
- Twitter

        limit: default: all there is
count limit for result
        offset: default: 0
offset for the result data
        
    """
    url = f"https://k-pop.p.rapidapi.com/idols"
    querystring = {'q': q, }
    if by:
        querystring['by'] = by
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "k-pop.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

