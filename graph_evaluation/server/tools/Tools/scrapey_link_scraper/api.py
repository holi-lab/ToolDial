import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def scrape_links(url: str, maxlinks: int=10, includequery: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scrape all links from URL"
    
    """
    url = f"https://scrapey-link-scraper.p.rapidapi.com/v1/scrapelinks/"
    querystring = {'url': url, }
    if maxlinks:
        querystring['maxlinks'] = maxlinks
    if includequery:
        querystring['includequery'] = includequery
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrapey-link-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

