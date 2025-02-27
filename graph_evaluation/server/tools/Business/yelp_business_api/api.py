import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def each_business_scrape_by_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scrape By Yelp URL. Ex. https://www.yelp.com/biz/capital-blossom-day-spa-washington
		
		You can get these business urls from the "/search" endpoint('YelpURL')"
    
    """
    url = f"https://yelp-business-api.p.rapidapi.com/eachbusiness"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yelp-business-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(location: str, page: str, query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search any business type in any location. Ex. Coffee Shop, New York"
    
    """
    url = f"https://yelp-business-api.p.rapidapi.com/search"
    querystring = {'location': location, 'page': page, 'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yelp-business-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

