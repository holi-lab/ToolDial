import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def pursuit(q: str, filetype: str=None, type: str=None, sort: str=None, start: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a query in file names and path"
    q: Search Query 
        filetype: File Extension, only one file extension accepted per query
        type: Type of files (Accepted values: video, audio, ebook, archive, mobile)
        sort: Sort by ascending or descending values (Accepted values: sizeasc, sizedesc, dateasc, datedesc, fileasc, filedesc), only one accepted per query. By default datedesc is selected.
        start: Start results from count (for pagination)
        
    """
    url = f"https://filepursuit.p.rapidapi.com/"
    querystring = {'q': q, }
    if filetype:
        querystring['filetype'] = filetype
    if type:
        querystring['type'] = type
    if sort:
        querystring['sort'] = sort
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "filepursuit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def discover(start: int=None, type: str=None, link: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Discover every file crawled by FilePursuit using links and domains"
    start: Start results from count (for pagination)
        type: Type of result (Accepted values: filelinks, all)
        link: Link you wish to discover 
        
    """
    url = f"https://filepursuit.p.rapidapi.com/discover"
    querystring = {}
    if start:
        querystring['start'] = start
    if type:
        querystring['type'] = type
    if link:
        querystring['link'] = link
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "filepursuit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

