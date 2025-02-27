import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_product_reviews(count: str, page: str, app_key: str, domain_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get all reviews for a specific product that belongs to a specific app_key"
    count: specify how many reviews you want to pull
        page: specify the page number you want to pull
        app_key: your application key that is assigned to you after signing up to Yotpo (www.yotpo.com)
        domain_key: unique identifier of the product as stored when creating the review
        
    """
    url = f"https://yotpo.p.rapidapi.com/products/{app_key}/{domain_key}/reviews"
    querystring = {'count': count, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yotpo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

