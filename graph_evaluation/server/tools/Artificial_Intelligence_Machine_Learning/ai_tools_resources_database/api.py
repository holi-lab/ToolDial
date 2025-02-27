import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def news(time: str='This Month', category: str=None, page: str='1', sort: str='Popular', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ai news/articles"
    time: All Time | This Week | This Month | Today
        category: All | Updates | Interesting | Video | Podcast | Learn | Research | Opinion
        sort: New | Featured | Popular
        
    """
    url = f"https://ai-tools-resources-database.p.rapidapi.com/getPosts"
    querystring = {}
    if time:
        querystring['time'] = time
    if category:
        querystring['category'] = category
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-tools-resources-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tags(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get all AI tools tags"
    
    """
    url = f"https://ai-tools-resources-database.p.rapidapi.com/tags"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-tools-resources-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tools_apps(tag: str=None, sort: str='new', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest AI apps and tools
		sort: new | popular | verified"
    sort: new | popular | verified
        
    """
    url = f"https://ai-tools-resources-database.p.rapidapi.com/tools"
    querystring = {}
    if tag:
        querystring['tag'] = tag
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-tools-resources-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

