import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def brands(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "brands"
    
    """
    url = f"https://sneakers-real-time-pricing.p.rapidapi.com/sneakers/brands"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sneakers-real-time-pricing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def platforms(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "platforms"
    
    """
    url = f"https://sneakers-real-time-pricing.p.rapidapi.com/sneakers/platforms"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sneakers-real-time-pricing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prices_stats(sku: str, size_uk: str=None, size_eu: str=None, size_jp: str=None, size_us: str='6', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "prices stats"
    
    """
    url = f"https://sneakers-real-time-pricing.p.rapidapi.com/sneakers/prices_stats"
    querystring = {'sku': sku, }
    if size_uk:
        querystring['size_uk'] = size_uk
    if size_eu:
        querystring['size_eu'] = size_eu
    if size_jp:
        querystring['size_jp'] = size_jp
    if size_us:
        querystring['size_us'] = size_us
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sneakers-real-time-pricing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(rekeased_at: str=None, brand_id: str=None, extended: str='true', sizing: str=None, is_id: str=None, sku: str=None, colorway: str=None, initial_price: int=None, name: str='jordan', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get product details"
    rekeased_at: date of the release
        brand_id: query with the brand id
        extended: show more data
        sizing: nature of the sneaker
        is_id: internal id
        sku: sku of the sneaker
        colorway: color of the sneaker
        initial_price: price of the sneaker
        name: name of the sneaker
        
    """
    url = f"https://sneakers-real-time-pricing.p.rapidapi.com/sneakers"
    querystring = {}
    if rekeased_at:
        querystring['rekeased_at'] = rekeased_at
    if brand_id:
        querystring['brand_id'] = brand_id
    if extended:
        querystring['extended'] = extended
    if sizing:
        querystring['sizing'] = sizing
    if is_id:
        querystring['id'] = is_id
    if sku:
        querystring['sku'] = sku
    if colorway:
        querystring['colorway'] = colorway
    if initial_price:
        querystring['initial_price'] = initial_price
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sneakers-real-time-pricing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product(kikikickz_id: str=None, restocks_id: str=None, stockx_slug: str=None, brand_id: str=None, released_at: str=None, sku: str=None, laced_id: str=None, stockx_id: str=None, hypeboost_id: str=None, klekt_slug: str=None, colorway: str=None, klekt_id: str=None, hypeboost_slug: str=None, is_id: str=None, sizing: str=None, initial_price: str=None, kikikickz_slug: str=None, alias_slug: str=None, laced_slug: str=None, restocks_slug: str=None, name: str=None, alias_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Details of a product"
    kikikickz_id: if you have the external id, you can use this argument
        restocks_id: if you have the external id, you can use this argument
        stockx_slug: if you have the external id, you can use this argument
        brand_id: brand internal id
        released_at: date of the release
        sku: sku of the sneaker
        laced_id: if you have the external id, you can use this argument
        stockx_id: if you have the external id, you can use this argument
        hypeboost_id: if you have the external id, you can use this argument
        klekt_slug: if you have the external id, you can use this argument
        colorway: color of the sneaker
        klekt_id: if you have the external id, you can use this argument
        hypeboost_slug: if you have the external id, you can use this argument
        is_id: internal id
        sizing: nature of the sneaker
        initial_price: initial price
        kikikickz_slug: if you have the external id, you can use this argument
        alias_slug: if you have the external id, you can use this argument
        laced_slug: if you have the external id, you can use this argument
        restocks_slug: if you have the external id, you can use this argument
        name: name of the sneaker
        alias_id: if you have the external id, you can use this argument
        
    """
    url = f"https://sneakers-real-time-pricing.p.rapidapi.com/sneakers/product"
    querystring = {}
    if kikikickz_id:
        querystring['kikikickz_id'] = kikikickz_id
    if restocks_id:
        querystring['restocks_id'] = restocks_id
    if stockx_slug:
        querystring['stockx_slug'] = stockx_slug
    if brand_id:
        querystring['brand_id'] = brand_id
    if released_at:
        querystring['released_at'] = released_at
    if sku:
        querystring['sku'] = sku
    if laced_id:
        querystring['laced_id'] = laced_id
    if stockx_id:
        querystring['stockx_id'] = stockx_id
    if hypeboost_id:
        querystring['hypeboost_id'] = hypeboost_id
    if klekt_slug:
        querystring['klekt_slug'] = klekt_slug
    if colorway:
        querystring['colorway'] = colorway
    if klekt_id:
        querystring['klekt_id'] = klekt_id
    if hypeboost_slug:
        querystring['hypeboost_slug'] = hypeboost_slug
    if is_id:
        querystring['id'] = is_id
    if sizing:
        querystring['sizing'] = sizing
    if initial_price:
        querystring['initial_price'] = initial_price
    if kikikickz_slug:
        querystring['kikikickz_slug'] = kikikickz_slug
    if alias_slug:
        querystring['alias_slug'] = alias_slug
    if laced_slug:
        querystring['laced_slug'] = laced_slug
    if restocks_slug:
        querystring['restocks_slug'] = restocks_slug
    if name:
        querystring['name'] = name
    if alias_id:
        querystring['alias_id'] = alias_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sneakers-real-time-pricing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prices(size_us: int, platform_name: str, product_id: str, product_sku: str, is_id: str, platform_id: str, size_uk: int, size_eu: int, size_jp: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "the most interesting section. All sneakers prices through marketplaces."
    size_us: size in usa
        platform_name: the name of the plateform (stockx, laced...)
        product_id: internal product id
        product_sku: product sku
        id: internal id
        platform_id: internal platform id
        size_uk: size in uk
        size_eu: size in europe
        size_jp: size in japan
        
    """
    url = f"https://sneakers-real-time-pricing.p.rapidapi.com/sneakers/prices"
    querystring = {'size_us': size_us, 'platform_name': platform_name, 'product_id': product_id, 'product_sku': product_sku, 'id': is_id, 'platform_id': platform_id, 'size_uk': size_uk, 'size_eu': size_eu, 'size_jp': size_jp, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sneakers-real-time-pricing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

