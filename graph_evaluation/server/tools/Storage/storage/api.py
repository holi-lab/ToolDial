import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getshareablelink(entryid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    entryid: ID or hash of entry for which to get shareable link
        
    """
    url = f"https://storage5.p.rapidapi.com/entries/{entryid}/shareable-link"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "storage5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def indexentry(starredonly: bool=None, query: str=None, type: str='folder', sharedonly: bool=None, recentonly: bool=None, parentids: str='[]', deletedonly: bool=None, perpage: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of all file entries you have access to"
    starredonly: Whether only starred entries should be returned
        query: Search query entry names should be filtered on
        type: File type entries should be filtered on
        sharedonly: Whether only entries that have been shared with you should be returned
        recentonly: Whether only recent entries should be returned
        parentids: Only entries that are children of specified folders will be returned
        deletedonly: Whether only trashed entries should be returned
        perpage: How many entries to return per page
        
    """
    url = f"https://storage5.p.rapidapi.com/entries"
    querystring = {}
    if starredonly:
        querystring['starredOnly'] = starredonly
    if query:
        querystring['query'] = query
    if type:
        querystring['type'] = type
    if sharedonly:
        querystring['sharedOnly'] = sharedonly
    if recentonly:
        querystring['recentOnly'] = recentonly
    if parentids:
        querystring['parentIds'] = parentids
    if deletedonly:
        querystring['deletedOnly'] = deletedonly
    if perpage:
        querystring['perPage'] = perpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "storage5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

