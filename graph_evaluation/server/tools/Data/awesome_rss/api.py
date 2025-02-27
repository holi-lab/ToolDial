import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_feeds(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List your current RSS"
    
    """
    url = f"https://awesome-rss.p.rapidapi.com/v1/feeds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "awesome-rss.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_feedid(feedid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get RSS content
		
		**If you use an RSS reader**
		
		If you use RapidAPI, you will need to include two header parameters, *X-RapidAPI-Key* and *X-RapidAPI-Host*.
		
		I know this may not be friendly to RSS readers, so I suggest you use `https://awesome-rss.cys.fyi/api/v1/{feedId}` instead to get your content. Then add the link to your reader.
		"
    feedid: FeedID will be converted to slugs, e.g. foo_bar > foo-bar.

Each user can generate 5 RSS, and each RSS can have up to 20 items.

        
    """
    url = f"https://awesome-rss.p.rapidapi.com/v1/{feedid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "awesome-rss.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

