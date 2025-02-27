import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product_reviews(page: str, url: str='https://www.walmart.com/ip/Sceptre-65-Class-4K-UHD-LED-TV-HDR-U650CV-U/48874705', usitemid: str='48874705', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get product reviews on Walmart.com by specifying product page url or usItemId."
    
    """
    url = f"https://walmart3.p.rapidapi.com/product-reviews"
    querystring = {'page': page, }
    if url:
        querystring['url'] = url
    if usitemid:
        querystring['usItemId'] = usitemid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walmart3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def store_location(zip_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed Walmart store locations (phone number and address) by specifying the zip code."
    
    """
    url = f"https://walmart3.p.rapidapi.com/store-location"
    querystring = {'zip_code': zip_code, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "walmart3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

