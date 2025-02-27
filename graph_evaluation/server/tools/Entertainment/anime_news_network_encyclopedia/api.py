import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reports(is_id: int, nlist: int=50, nskip: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    id: ID of the report to display; list of available reports is at http://www.animenewsnetwork.com/encyclopedia/reports.php
        nlist: limit number of results in the report
        nskip: skip the the first N items in the report
        
    """
    url = f"https://animenewsnetwork.p.rapidapi.com/reports.xml"
    querystring = {'id': is_id, }
    if nlist:
        querystring['nlist'] = nlist
    if nskip:
        querystring['nskip'] = nskip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animenewsnetwork.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def anime_manga_details(title: int=None, anime: int=None, manga: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    title: ID of the anime/manga
        anime: ID of the anime
        manga: ID of the manga
        
    """
    url = f"https://animenewsnetwork.p.rapidapi.com/api.xml"
    querystring = {}
    if title:
        querystring['title'] = title
    if anime:
        querystring['anime'] = anime
    if manga:
        querystring['manga'] = manga
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animenewsnetwork.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

