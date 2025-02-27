import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_1688_package_details(storeid: str, itemid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint shows the package details for this specific item. It shows its' dimensions such as height, length, and width. Also shows the estimated Weight."
    
    """
    url = f"https://16881.p.rapidapi.com/package_detail"
    querystring = {'storeId': storeid, 'itemId': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "16881.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_search_by_image(imgurl: str, page: int=1, catid: str=None, sort: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Imgs"
    
    """
    url = f"https://16881.p.rapidapi.com/item_search_image"
    querystring = {'imgUrl': imgurl, }
    if page:
        querystring['page'] = page
    if catid:
        querystring['catId'] = catid
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "16881.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_search(q: str, api: str, page: int=1, sort: str='default', page_size: int=20, start_price: int=None, end_price: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search products from  1688 website"
    q: Search query
        page: Page number
        sort: Item sorting by default (best match), price and sales. Choose from available options.
        page_size: Page Size
        start_price: Starting Low Price
        end_price: Ending High Price
        
    """
    url = f"https://16881.p.rapidapi.com/api"
    querystring = {'q': q, 'api': api, }
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if page_size:
        querystring['page_size'] = page_size
    if start_price:
        querystring['start_price'] = start_price
    if end_price:
        querystring['end_price'] = end_price
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "16881.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_details(num_iid: int, api: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Details of a single item by item ID"
    num_iid: Item ID
        api: Do Not Edit This Field
        
    """
    url = f"https://16881.p.rapidapi.com/api"
    querystring = {'num_iid': num_iid, 'api': api, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "16881.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

