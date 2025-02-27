import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_full_text_jsonfeed_content_from_url(url: str, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method not only parses the RSS/ATOM/RDF feeds but also full-text parses the item pages that are linked to."
    
    """
    url = f"https://rss-atom-rdf-to-jsonfeed.p.rapidapi.com/feed-full"
    querystring = {'url': url, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rss-atom-rdf-to-jsonfeed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_jsonfeed_format_from_url(url: str, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request metadata about the feed domain and a list of articles from the url source formatted in JSON following JSONFeed format."
    url: The url of the feed target you want to extract the JSON from. Mandatory with http/s protocol.
        limit: Limit of Articles returned by the feed. Max 50. Default 10.
        
    """
    url = f"https://rss-atom-rdf-to-jsonfeed.p.rapidapi.com/feed"
    querystring = {'url': url, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rss-atom-rdf-to-jsonfeed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

