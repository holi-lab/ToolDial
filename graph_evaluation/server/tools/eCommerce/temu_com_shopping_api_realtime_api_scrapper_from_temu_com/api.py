import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product_details(goodsid: str, content_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the product details information based on the given goodsId (product id defined by temu, returned by the /search endpoint)
		Response includes:
		- price information
		- product reviews
		- offers/availability
		- sku - variants in different options like size, color etc.
		- metadata, image gallery, description ...
		
		-"
    
    """
    url = f"https://temu-com-shopping-api-realtime-api-scrapper-from-temu-com.p.rapidapi.com/details"
    querystring = {'goodsId': goodsid, }
    if content_type:
        querystring['Content-Type'] = content_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "temu-com-shopping-api-realtime-api-scrapper-from-temu-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_search(keyword: str, content_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for list of products by providing the keyword."
    
    """
    url = f"https://temu-com-shopping-api-realtime-api-scrapper-from-temu-com.p.rapidapi.com/search"
    querystring = {'keyword': keyword, }
    if content_type:
        querystring['Content-Type'] = content_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "temu-com-shopping-api-realtime-api-scrapper-from-temu-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

