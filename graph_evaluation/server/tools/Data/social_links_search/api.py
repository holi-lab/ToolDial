import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_social_links(query: str, social_networks: str='facebook,tiktok,instagram,snapchat,twitter,youtube,linkedin,github,pinterest', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get social profile links by search query or keywords. The following social networks are supported: Facebook, TikTok, Instagram, Snapchat, Twitter, Youtube, LinkedIn, GitHub and Pinterest."
    query: Social links search query.
        social_networks: Find social links for the specified social networks, specified as a comma delimited list of the following values: `facebook`, `tiktok`, `instagram`, `snapchat`, `twitter`, `youtube`, `linkedin`, `github`, `pinterest`.

**Default:** *facebook,tiktok,instagram,snapchat,twitter,youtube,linkedin,github*
        
    """
    url = f"https://social-links-search.p.rapidapi.com/search-social-links"
    querystring = {'query': query, }
    if social_networks:
        querystring['social_networks'] = social_networks
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "social-links-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

