import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_item_review(itemid: str, framesize: int, language: str, frameposition: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviews for Jd products"
    
    """
    url = f"https://otapi-jd.p.rapidapi.com/SearchItemReviews"
    querystring = {'ItemId': itemid, 'frameSize': framesize, 'language': language, 'framePosition': frameposition, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-jd.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vendor(language: str='en', vendorid: str='jd-31572', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Jd seller information"
    
    """
    url = f"https://otapi-jd.p.rapidapi.com/GetVendorInfo"
    querystring = {}
    if language:
        querystring['language'] = language
    if vendorid:
        querystring['vendorId'] = vendorid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-jd.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_description(itemid: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Jd product description"
    
    """
    url = f"https://otapi-jd.p.rapidapi.com/GetItemDescription"
    querystring = {'itemId': itemid, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-jd.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item(itemid: str='jd-10030391539604', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full data from Jd product by Item Id"
    
    """
    url = f"https://otapi-jd.p.rapidapi.com/BatchGetItemFullInfo"
    querystring = {}
    if itemid:
        querystring['itemId'] = itemid
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-jd.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_items(frameposition: int, language: str, framesize: int, imageurl: str='https://img.alicdn.com/bao/uploaded/i4/2773693278/O1CN01YrlRK91a5MwEi1osq_!!2773693278.jpg_500x500.jpg', minprice: int=100, itemtitle: str='glasses', minvolume: int=50, maxprice: int=5000, categoryid: str='jd-9922', orderby: str='Popularity:Desc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Jd products by title, price, image, orders volume."
    
    """
    url = f"https://otapi-jd.p.rapidapi.com/BatchSearchItemsFrame"
    querystring = {'framePosition': frameposition, 'language': language, 'frameSize': framesize, }
    if imageurl:
        querystring['ImageUrl'] = imageurl
    if minprice:
        querystring['MinPrice'] = minprice
    if itemtitle:
        querystring['ItemTitle'] = itemtitle
    if minvolume:
        querystring['MinVolume'] = minvolume
    if maxprice:
        querystring['MaxPrice'] = maxprice
    if categoryid:
        querystring['CategoryId'] = categoryid
    if orderby:
        querystring['OrderBy'] = orderby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-jd.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_catalog(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full catalog for Jd"
    
    """
    url = f"https://otapi-jd.p.rapidapi.com/GetBriefCatalog"
    querystring = {'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-jd.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

