import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def image_search(keyword: str, order: str='title', size: int=10, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search tour images by keyword you input."
    keyword: Keyword should be two characters at least.
        order: Order parameter should be \\\\\\\"new\\\\\\\", \\\\\\\"title\\\\\\\", or \\\\\\\"count\\\\\\\".
        size: Default 10
        page: Default 1
        
    """
    url = f"https://korean-tour-gallery.p.rapidapi.com/search"
    querystring = {'keyword': keyword, }
    if order:
        querystring['order'] = order
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "korean-tour-gallery.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def image_list(size: int=10, page: int=1, order: str='title', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tour image list by default."
    size: Default 10
        page: Default 1
        order: Order parameter should be \\\\\\\"new\\\\\\\", \\\\\\\"title\\\\\\\", or \\\\\\\"count\\\\\\\".
        
    """
    url = f"https://korean-tour-gallery.p.rapidapi.com/photos"
    querystring = {}
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "korean-tour-gallery.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

