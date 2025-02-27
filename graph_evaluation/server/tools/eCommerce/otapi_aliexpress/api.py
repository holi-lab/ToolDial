import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_item(itemid: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Aliexpress product full information"
    
    """
    url = f"https://otapi-aliexpress.p.rapidapi.com/BatchGetItemFullInfo"
    querystring = {'itemId': itemid, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-aliexpress.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_items(framesize: int, frameposition: int, language: str, itemtitle: str='glasses', maxprice: int=5000, imageurl: str='https://img.alicdn.com/bao/uploaded/i4/2773693278/O1CN01YrlRK91a5MwEi1osq_!!2773693278.jpg_500x500.jpg', minprice: int=100, orderby: str='Popularity:Desc', minvolume: int=50, categoryid: str='ae-100000162', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Aliexpress products by title, price, image, orders volume."
    
    """
    url = f"https://otapi-aliexpress.p.rapidapi.com/BatchSearchItemsFrame"
    querystring = {'frameSize': framesize, 'framePosition': frameposition, 'language': language, }
    if itemtitle:
        querystring['ItemTitle'] = itemtitle
    if maxprice:
        querystring['MaxPrice'] = maxprice
    if imageurl:
        querystring['ImageUrl'] = imageurl
    if minprice:
        querystring['MinPrice'] = minprice
    if orderby:
        querystring['OrderBy'] = orderby
    if minvolume:
        querystring['MinVolume'] = minvolume
    if categoryid:
        querystring['CategoryId'] = categoryid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-aliexpress.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vendor(language: str='en', vendorid: str='ae-246807196', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Aliexpress seller information"
    
    """
    url = f"https://otapi-aliexpress.p.rapidapi.com/GetVendorInfo"
    querystring = {}
    if language:
        querystring['language'] = language
    if vendorid:
        querystring['vendorId'] = vendorid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-aliexpress.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_review(language: str='en', itemid: str='ae-4000250369249', framesize: str='50', frameposition: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviews for Aliexpress products"
    
    """
    url = f"https://otapi-aliexpress.p.rapidapi.com/SearchItemReviews"
    querystring = {}
    if language:
        querystring['language'] = language
    if itemid:
        querystring['ItemId'] = itemid
    if framesize:
        querystring['frameSize'] = framesize
    if frameposition:
        querystring['framePosition'] = frameposition
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-aliexpress.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_description(language: str, itemid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Aliexpress product description"
    
    """
    url = f"https://otapi-aliexpress.p.rapidapi.com/GetItemDescription"
    querystring = {'language': language, 'itemId': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-aliexpress.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_catalog(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full catalog for Aliexpress"
    
    """
    url = f"https://otapi-aliexpress.p.rapidapi.com/GetBriefCatalog"
    querystring = {'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-aliexpress.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

