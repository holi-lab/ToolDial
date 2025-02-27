import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def links(links: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "single Links ( see list host support from http://urlchecker.net/#hosts) (rapidgator.net,mediafire.com...)"
    links: single Links ( see list host support from http://urlchecker.net/#hosts) (rapidgator.net,mediafire.com...)
        
    """
    url = f"https://urlchecker.p.rapidapi.com/"
    querystring = {'links': links, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "urlchecker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def decodelinks(decodelinks: str, response_format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Decode multi link or website"
    decodelinks: Decode multi link or website
        response_format: 'xml' ,'txt' or 'json' (default 'xml')
        
    """
    url = f"https://urlchecker.p.rapidapi.com/"
    querystring = {'decodelinks': decodelinks, }
    if response_format:
        querystring['response_format'] = response_format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "urlchecker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def decodelink(decodelink: str, response_format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Decode single link only (adf.ly,safelinking.net,bit.ly ...)"
    decodelink: Decode single link only (adf.ly,safelinking.net,bit.ly ...)
        response_format: 'xml' ,'txt' or 'json' (default 'xml')
        
    """
    url = f"https://urlchecker.p.rapidapi.com/"
    querystring = {'decodelink': decodelink, }
    if response_format:
        querystring['response_format'] = response_format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "urlchecker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def url(url: str, response_format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "website or folder need extract ( extract and check )"
    url: website or folder need extract ( extract and check )
        response_format: 'xml' ,'txt' or 'json' (default 'xml')
        
    """
    url = f"https://urlchecker.p.rapidapi.com/"
    querystring = {'url': url, }
    if response_format:
        querystring['response_format'] = response_format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "urlchecker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

