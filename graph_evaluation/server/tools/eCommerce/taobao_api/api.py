import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def item_details_simple(num_iid: str, api: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using "item ID"  or "item string ID "get product details."
    num_iid: Taobao/Tmall Item ID or String ID.

Use \"Item ID\" or \"Item String ID \"to get product details.
        api: Don’t edit this field
        
    """
    url = f"https://taobao-api.p.rapidapi.com/api"
    querystring = {'num_iid': num_iid, 'api': api, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def items_search(api: str, loc: str=None, cat: int=None, page: int=None, q: str='shoes', end_price: int=None, free_shiping: bool=None, tmall: bool=None, page_size: int=40, sort: str=None, start_price: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Real-time search for products using query or category. Then filter and sort the results."
    api: Don’t edit this field
        loc: Name of the city in China (in Chinese)
        cat: Category ID
        page: Page Number
        q: Search Query
        end_price: Filter by Ending Price
        free_shiping: Is free shipping (only in Mailand China)
        tmall: Taobao, Tmall all products / Tmall products
        page_size: Result returned items quantity
        sort: Item sorting: default,sales_asc,sales_des,price_asc,price_des
        start_price: Filter by Starting Price
        
    """
    url = f"https://taobao-api.p.rapidapi.com/api"
    querystring = {'api': api, }
    if loc:
        querystring['loc'] = loc
    if cat:
        querystring['cat'] = cat
    if page:
        querystring['page'] = page
    if q:
        querystring['q'] = q
    if end_price:
        querystring['end_price'] = end_price
    if free_shiping:
        querystring['free_shiping'] = free_shiping
    if tmall:
        querystring['tmall'] = tmall
    if page_size:
        querystring['page_size'] = page_size
    if sort:
        querystring['sort'] = sort
    if start_price:
        querystring['start_price'] = start_price
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_description(api: str, num_iid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using item ID get product Description Images."
    api: Don’t edit this field
        num_iid: Item ID
        
    """
    url = f"https://taobao-api.p.rapidapi.com/api"
    querystring = {'api': api, 'num_iid': num_iid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_sku(api: str, num_iid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using item ID get product SKU information."
    api: Don’t edit this field
        num_iid: Item ID
        
    """
    url = f"https://taobao-api.p.rapidapi.com/api"
    querystring = {'api': api, 'num_iid': num_iid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_delivery_fee(api: str, num_iid: int, view_skus: bool=None, area_id: int=110000, sku_id: str='default', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get item delivery Fee"
    api: Do Not Edit This Field.
        view_skus: Shows delivery fees for every SKU. Default value \\\"false\\\".
        area_id: Area ID. Area ID can be obtained via areas_get API. Default value \\\"110000\\\".
        sku_id: Item SKU ID. Default value \\\"default\\\".
        
    """
    url = f"https://taobao-api.p.rapidapi.com/api"
    querystring = {'api': api, 'num_iid': num_iid, }
    if view_skus:
        querystring['view_skus'] = view_skus
    if area_id:
        querystring['area_id'] = area_id
    if sku_id:
        querystring['sku_id'] = sku_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shops_search_simple(q: str, api: str, sort: str=None, is_tmall: bool=None, end_credit: int=None, page: int=1, start_credit: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for different shops using query. Then filter and sort the results."
    q: Search Query
        api: Don’t edit this field
        sort: Result sort
        is_tmall: Taobao, Tmall all products / Tmall products
        end_credit: Seller credit end
        page: Page Number
        start_credit: Seller credit start
        page_size: Result returned items quantity
        
    """
    url = f"https://taobao-api.p.rapidapi.com/api"
    querystring = {'q': q, 'api': api, }
    if sort:
        querystring['sort'] = sort
    if is_tmall:
        querystring['is_tmall'] = is_tmall
    if end_credit:
        querystring['end_credit'] = end_credit
    if page:
        querystring['page'] = page
    if start_credit:
        querystring['start_credit'] = start_credit
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

