import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def categories(locale: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aliexpress-unofficial.p.rapidapi.com/categories"
    querystring = {}
    if locale:
        querystring['locale'] = locale
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feedbacks_id(is_id: int, withpictures: int=1, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aliexpress-unofficial.p.rapidapi.com/feedbacks/{is_id}"
    querystring = {}
    if withpictures:
        querystring['withPictures'] = withpictures
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_id(is_id: int, locale: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aliexpress-unofficial.p.rapidapi.com/categories/{is_id}"
    querystring = {}
    if locale:
        querystring['locale'] = locale
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shipping_id(is_id: int, count: int=1, locale: str=None, country: str=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aliexpress-unofficial.p.rapidapi.com/shipping/{is_id}"
    querystring = {}
    if count:
        querystring['count'] = count
    if locale:
        querystring['locale'] = locale
    if country:
        querystring['country'] = country
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_id(is_id: int, page: int=1, currency: str=None, sort: str=None, pricemax: int=100, country: str=None, pricemin: int=20, issale: bool=None, locale: str=None, isfreeship: bool=None, isfavorite: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aliexpress-unofficial.p.rapidapi.com/products/{is_id}"
    querystring = {}
    if page:
        querystring['page'] = page
    if currency:
        querystring['currency'] = currency
    if sort:
        querystring['sort'] = sort
    if pricemax:
        querystring['priceMax'] = pricemax
    if country:
        querystring['country'] = country
    if pricemin:
        querystring['priceMin'] = pricemin
    if issale:
        querystring['isSale'] = issale
    if locale:
        querystring['locale'] = locale
    if isfreeship:
        querystring['isFreeShip'] = isfreeship
    if isfavorite:
        querystring['isFavorite'] = isfavorite
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_id(is_id: int, locale: str=None, currency: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aliexpress-unofficial.p.rapidapi.com/product/{is_id}"
    querystring = {}
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aliexpress-unofficial.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

