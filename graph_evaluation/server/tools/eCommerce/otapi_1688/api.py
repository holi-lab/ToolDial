import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_vendor(language: str, vendorid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 1688 seller information"
    
    """
    url = f"https://otapi-1688.p.rapidapi.com/GetVendorInfo"
    querystring = {'language': language, 'vendorId': vendorid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-1688.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_items(language: str, frameposition: int, framesize: int, minvolume: int=50, imageurl: str='https://img.alicdn.com/bao/uploaded/i4/2773693278/O1CN01YrlRK91a5MwEi1osq_!!2773693278.jpg_500x500.jpg', categoryid: str='abb-156', minprice: int=100, itemtitle: str='glasses', orderby: str='Popularity:Desc', maxprice: int=5000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for 1688 products by title, price, image, orders volume."
    
    """
    url = f"https://otapi-1688.p.rapidapi.com/BatchSearchItemsFrame"
    querystring = {'language': language, 'framePosition': frameposition, 'frameSize': framesize, }
    if minvolume:
        querystring['MinVolume'] = minvolume
    if imageurl:
        querystring['ImageUrl'] = imageurl
    if categoryid:
        querystring['CategoryId'] = categoryid
    if minprice:
        querystring['MinPrice'] = minprice
    if itemtitle:
        querystring['ItemTitle'] = itemtitle
    if orderby:
        querystring['OrderBy'] = orderby
    if maxprice:
        querystring['MaxPrice'] = maxprice
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-1688.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_review(framesize: int, frameposition: int, itemid: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviews for 1688 products"
    
    """
    url = f"https://otapi-1688.p.rapidapi.com/SearchItemReviews"
    querystring = {'frameSize': framesize, 'framePosition': frameposition, 'ItemId': itemid, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-1688.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_description(language: str, itemid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 1688 product description"
    
    """
    url = f"https://otapi-1688.p.rapidapi.com/GetItemDescription"
    querystring = {'language': language, 'itemId': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-1688.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item(itemid: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full data from 1688 product by Item Id"
    
    """
    url = f"https://otapi-1688.p.rapidapi.com/BatchGetItemFullInfo"
    querystring = {'itemId': itemid, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-1688.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_catalog(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full catalog for 1688"
    
    """
    url = f"https://otapi-1688.p.rapidapi.com/GetBriefCatalog"
    querystring = {'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-1688.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

