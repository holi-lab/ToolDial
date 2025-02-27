import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def deals(page: str, per_page: str, details: str='0', categories: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This shows deals"
    
    """
    url = f"https://ecommerce-scraper.p.rapidapi.com/api/posts"
    querystring = {'page': page, 'per_page': per_page, }
    if details:
        querystring['details'] = details
    if categories:
        querystring['categories'] = categories
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecommerce-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

