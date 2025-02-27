import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_item_description(itemid: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Amazon product description"
    
    """
    url = f"https://otapi-amazon.p.rapidapi.com/GetItemDescription"
    querystring = {'itemId': itemid, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-amazon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_items(frameposition: int, framesize: int, language: str, categoryid: str='az-2972638011', minprice: int=10, itemtitle: str='patio', maxprice: int=800, orderby: str='Popularity:Desc', imageurl: str='https://m.media-amazon.com/images/I/81Dky9R4+pL._AC_UL960_QL65_.jpg', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Amazon products by title, price, image, orders volume."
    
    """
    url = f"https://otapi-amazon.p.rapidapi.com/BatchSearchItemsFrame"
    querystring = {'framePosition': frameposition, 'frameSize': framesize, 'language': language, }
    if categoryid:
        querystring['CategoryId'] = categoryid
    if minprice:
        querystring['MinPrice'] = minprice
    if itemtitle:
        querystring['ItemTitle'] = itemtitle
    if maxprice:
        querystring['MaxPrice'] = maxprice
    if orderby:
        querystring['OrderBy'] = orderby
    if imageurl:
        querystring['ImageUrl'] = imageurl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-amazon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vendor(language: str, vendorid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Amazon seller information"
    
    """
    url = f"https://otapi-amazon.p.rapidapi.com/GetVendorInfo"
    querystring = {'language': language, 'vendorId': vendorid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-amazon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item(itemid: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full data from Amazon  product by Item Id"
    
    """
    url = f"https://otapi-amazon.p.rapidapi.com/BatchGetItemFullInfo"
    querystring = {'itemId': itemid, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-amazon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_review(framesize: int, frameposition: int, language: str, itemid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviews for Amazon products"
    
    """
    url = f"https://otapi-amazon.p.rapidapi.com/SearchItemReviews"
    querystring = {'frameSize': framesize, 'framePosition': frameposition, 'language': language, 'ItemId': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-amazon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_catalog(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full catalog for Amazon"
    
    """
    url = f"https://otapi-amazon.p.rapidapi.com/GetBriefCatalog"
    querystring = {'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-amazon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

