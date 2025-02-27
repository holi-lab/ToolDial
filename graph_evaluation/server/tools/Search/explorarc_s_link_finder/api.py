import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def links_finder(query: str='roadmap', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ExplorArc's Link Finder API simplifies the process of finding relevant links by returning results based on a given query. With this powerful tool, users can easily access the information they need to streamline their workflow and achieve their goals"
    
    """
    url = f"https://explorarcs-link-finder.p.rapidapi.com/api/on_query_links"
    querystring = {}
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "explorarcs-link-finder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

