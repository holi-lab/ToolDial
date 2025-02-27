import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def bestbuyproductdata(keyword: str, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It will accept two parameters keyword and page.
		For Ex:
		keyword: iphone  (it is the search term and it can be anything for example: iphone, ps5, sony tv etc)
		page: min 1 (max depends on the number of products available)"
    
    """
    url = f"https://bestbuy-product-data.p.rapidapi.com/bestbuy/"
    querystring = {'keyword': keyword, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bestbuy-product-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

