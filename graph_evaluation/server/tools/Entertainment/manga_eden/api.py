import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def manga_list(language: str, p: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "- dictionaries in the key "manga" contains the manga's image ("im"), title ("t"), ID ("i"), alias ("a"), status ("s"), last chapter date ("ld"), hits ("h")  - "page", "start", "end" and "total" are self explanatory"
    language: Where [language] can be 0 (English) or 1 (Italian)
        p: Same as above but returns only 500 manga's informations (from manga X*500 to (X+1)*500, where X is the page fetched from the GET parameter 'p')
        
    """
    url = f"https://community-manga-eden.p.rapidapi.com/list/{language}"
    querystring = {}
    if p:
        querystring['p'] = p
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-manga-eden.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manga_info_and_chapters_list(manga_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returned data: all the informations and chapters of the manga."
    manga_id: Where [manga.id] is the manga's id you can get with the previous api call
        
    """
    url = f"https://community-manga-eden.p.rapidapi.com/manga/{manga_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-manga-eden.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

