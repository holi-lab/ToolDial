import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def content_source_section_json(section: str, source: str, limit: int=20, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    section: Limits the set of items by a section.
all | A section name

 To get all sections, specify all. To get a particular section, use the section name returned by this request:
 http://api.nytimes.com/svc/news/v3/content/section-list.json

        source: Limits the set of items by originating source.

all = items from both The New York Times and The International New York Times
nyt = New York Times items only
inyt = International New York Times items only (FKA The International Herald Tribune)

        limit: Limits the number of results.  Set to increment of 20 up to 500.
        offset: Sets the starting point of the result set.
        
    """
    url = f"https://ny-times-times-wire.p.rapidapi.com/content/{source}/{section}.json"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-times-wire.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def content_section_list_json(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://ny-times-times-wire.p.rapidapi.com/content/section-list.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-times-wire.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def content_json(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    url: The complete URL of a specific news item, URL-encoded or backslash-escaped
        
    """
    url = f"https://ny-times-times-wire.p.rapidapi.com/content.json"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ny-times-times-wire.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

