import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sourceurl_metadata(md5_of_source_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request more information about a source-url (URL where a hoster link was found)"
    md5_of_source_url: Request more information about a source-url (URL where a hoster link was found)
        
    """
    url = f"https://alluc-alluc-video-and-download-search-v1.p.rapidapi.com/api/sourcedata/{md5_of_source_url}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alluc-alluc-video-and-download-search-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hosterurl_filedata(filedataid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "request additional metadata about a filehost-url"
    filedataid: Filedataid can be retrieved using /api/search
        
    """
    url = f"https://alluc-alluc-video-and-download-search-v1.p.rapidapi.com/api/filedata/{filedataid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alluc-alluc-video-and-download-search-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def thumbnail(imageid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch a thumbnail. {imageid} is returned in /api/search/."
    imageid: Fetch a thumbnail. {imageid} is returned in /api/search/.
        
    """
    url = f"https://alluc-alluc-video-and-download-search-v1.p.rapidapi.com/api/thumbnail/{imageid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alluc-alluc-video-and-download-search-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(searchtype: str, is_from: int=0, count: int=10, getmeta: int=0, query: str='big+buck+bunny', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for download links using the usual alluc search operators(https://www.alluc.com/about/)."
    searchtype: "download" or "stream"
        is_from: where to start. For example if you want result 20-30, you will set count=10 and from=20
        count: max-amount of returned results. Can be between 1 - 100.
        getmeta: If you want additional info on hosterlinks and source, set this to 1. Only use if you really need it as it might make for slower queries.
        query: the search string. All alluc search operators can be used (https://www.alluc.com/about/). Make sure to urlencode this.
        
    """
    url = f"https://alluc-alluc-video-and-download-search-v1.p.rapidapi.com/api/search/{searchtype}"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if count:
        querystring['count'] = count
    if getmeta:
        querystring['getmeta'] = getmeta
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alluc-alluc-video-and-download-search-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

