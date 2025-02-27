import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def keyword_search(page: int, keyword: str, sortby: str='recommended', excludesponsored: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve keyword search results."
    
    """
    url = f"https://axesso-kaufland-data-service.p.rapidapi.com/kfl/kaufland-search-by-keyword"
    querystring = {'page': page, 'keyword': keyword, }
    if sortby:
        querystring['sortBy'] = sortby
    if excludesponsored:
        querystring['excludeSponsored'] = excludesponsored
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-kaufland-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_details(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query product details information."
    
    """
    url = f"https://axesso-kaufland-data-service.p.rapidapi.com/kfl/kaufland-lookup-product"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "axesso-kaufland-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

