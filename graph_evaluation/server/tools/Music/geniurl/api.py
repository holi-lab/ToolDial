import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_only_for_the_top_match(format: str=None, q: str='Lady Gaga - Born This Way', artist: str='Lady Gaga', song: str='Born This Way', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint searches only for the top match.
		
		It can be provided with either of these:
		- a `?q` parameter, which contains a query to search for
		- the parameters `?artist` and `&song` to get better results"
    
    """
    url = f"https://geniurl.p.rapidapi.com/search/top"
    querystring = {}
    if format:
        querystring['format'] = format
    if q:
        querystring['q'] = q
    if artist:
        querystring['artist'] = artist
    if song:
        querystring['song'] = song
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geniurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_best_matches(format: str=None, song: str='Bad Romance', q: str='Lady Gaga - Bad Romance', artist: str='Lady Gaga', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint searches for the 10 best matches.
		
		It can be provided with either of these:
		- a `?q` parameter, which contains a query to search for
		- the parameters `?artist` and `&song` to get better results"
    
    """
    url = f"https://geniurl.p.rapidapi.com/search"
    querystring = {}
    if format:
        querystring['format'] = format
    if song:
        querystring['song'] = song
    if q:
        querystring['q'] = q
    if artist:
        querystring['artist'] = artist
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geniurl.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

