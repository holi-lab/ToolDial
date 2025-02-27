import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def topics(callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of all topics"
    callback: Optional JSONP callback
        
    """
    url = f"https://uncovered-treasure-v1.p.rapidapi.com/topics"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uncovered-treasure-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def random(callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get random treasure"
    callback: Optional JSONP callback
        
    """
    url = f"https://uncovered-treasure-v1.p.rapidapi.com/random"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uncovered-treasure-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verse_verse(verse: str, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get treasure by verse"
    verse: Use common English name with chapter number and optionally verse, such as "1 Corinthians 1", "1 Corinthians 1:5" or "James 3". Does NOT support ranged searches, such as "James 3:1-4".
        callback: Optional JSONP callback
        
    """
    url = f"https://uncovered-treasure-v1.p.rapidapi.com/verse/{verse}"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uncovered-treasure-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_searchterm(searchterm: str, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for treasure by term, e.g. "wisdom""
    searchterm: e.g. "God's plan"
        callback: Optional JSONP callback
        
    """
    url = f"https://uncovered-treasure-v1.p.rapidapi.com/search/{searchterm}"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uncovered-treasure-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def today(callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get treasure of the day"
    callback: Optional JSONP callback
        
    """
    url = f"https://uncovered-treasure-v1.p.rapidapi.com/today"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uncovered-treasure-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topic_topic(topic: str, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get entries from a specific topic"
    topic: "seek God"
        callback: Optional JSONP callback
        
    """
    url = f"https://uncovered-treasure-v1.p.rapidapi.com/topic/{topic}"
    querystring = {}
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uncovered-treasure-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

