import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_for_items(language: str, frameposition: int, framesize: int, imageurl: str='https://img.alicdn.com/bao/uploaded/i4/2773693278/O1CN01YrlRK91a5MwEi1osq_!!2773693278.jpg_500x500.jpg', minprice: int=5, orderby: str='Popularity:Desc', categoryid: str='eb-67670', maxprice: int=1000, itemtitle: str='glasses', minvolume: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Ebay products by title, price, image, orders volume."
    
    """
    url = f"https://otapi-ebay.p.rapidapi.com/BatchSearchItemsFrame"
    querystring = {'language': language, 'framePosition': frameposition, 'frameSize': framesize, }
    if imageurl:
        querystring['ImageUrl'] = imageurl
    if minprice:
        querystring['MinPrice'] = minprice
    if orderby:
        querystring['OrderBy'] = orderby
    if categoryid:
        querystring['CategoryId'] = categoryid
    if maxprice:
        querystring['MaxPrice'] = maxprice
    if itemtitle:
        querystring['ItemTitle'] = itemtitle
    if minvolume:
        querystring['MinVolume'] = minvolume
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-ebay.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_review(framesize: int, frameposition: int, language: str, itemid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviews for Ebay products"
    
    """
    url = f"https://otapi-ebay.p.rapidapi.com/SearchItemReviews"
    querystring = {'frameSize': framesize, 'framePosition': frameposition, 'language': language, 'ItemId': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-ebay.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_description(itemid: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Ebay product description"
    
    """
    url = f"https://otapi-ebay.p.rapidapi.com/GetItemDescription"
    querystring = {'itemId': itemid, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-ebay.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vendor(vendorid: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Ebay seller information"
    
    """
    url = f"https://otapi-ebay.p.rapidapi.com/GetVendorInfo"
    querystring = {'vendorId': vendorid, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-ebay.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item(language: str, itemid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full data from Ebay product by Item Id"
    
    """
    url = f"https://otapi-ebay.p.rapidapi.com/BatchGetItemFullInfo"
    querystring = {'language': language, 'itemId': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-ebay.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_catalog(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full catalog for Ebay"
    
    """
    url = f"https://otapi-ebay.p.rapidapi.com/GetBriefCatalog"
    querystring = {'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-ebay.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

