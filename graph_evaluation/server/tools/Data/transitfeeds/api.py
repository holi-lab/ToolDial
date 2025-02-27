import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getfeedversions(feed: str='sfmta/60', page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of feed versions. This list can be filtered by a number of input parameters"
    feed: Restrict results to to the given feed ID
        page: The page of results to return
        
    """
    url = f"https://community-transitfeeds.p.rapidapi.com/getFeedVersions"
    querystring = {}
    if feed:
        querystring['feed'] = feed
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-transitfeeds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlatestfeedversion(feed: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    feed: The ID of the feed to download
        
    """
    url = f"https://community-transitfeeds.p.rapidapi.com/getLatestFeedVersion"
    querystring = {'feed': feed, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-transitfeeds.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

