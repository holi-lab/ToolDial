import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_aliexpress_by_image(imgid: str, sort: str=None, shipto: str='US', currency: str='USD', language: str='en_US', category: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You need to upload an image to get the image id [**imgId**]"
    
    """
    url = f"https://ecommerce-search.p.rapidapi.com/api/v1/aliexpress/search_by_image"
    querystring = {'imgId': imgid, }
    if sort:
        querystring['sort'] = sort
    if shipto:
        querystring['shipTo'] = shipto
    if currency:
        querystring['currency'] = currency
    if language:
        querystring['language'] = language
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecommerce-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_banggood_by_image(imgid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You need to upload an image to get the image id [**imgId**]"
    
    """
    url = f"https://ecommerce-search.p.rapidapi.com/api/v1/banggood/search_by_image"
    querystring = {'imgId': imgid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecommerce-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_dhgate_by_image(imgid: str, category: str=None, currency: str='USD', language: str='en', pagenum: str=None, shipto: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You need to upload an image to get the image id [**imgId**]"
    
    """
    url = f"https://ecommerce-search.p.rapidapi.com/api/v1/dhgate/search_by_image"
    querystring = {'imgId': imgid, }
    if category:
        querystring['category'] = category
    if currency:
        querystring['currency'] = currency
    if language:
        querystring['language'] = language
    if pagenum:
        querystring['pageNum'] = pagenum
    if shipto:
        querystring['shipTo'] = shipto
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecommerce-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upload_image_from_url(imgurl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Before you search for products with image, you need to upload an image to our server to get the image id [**imgId**]"
    
    """
    url = f"https://ecommerce-search.p.rapidapi.com/api/v1/upload"
    querystring = {'imgUrl': imgurl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ecommerce-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

