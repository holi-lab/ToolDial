import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_for_items(frameposition: int, framesize: int, language: str, imageurl: str='https://sc04.alicdn.com/kf/Uf02821f9aedf476a8817dcdc704774b4o.jpg', minprice: int=10, orderby: str='Popularity:Desc', minvolume: int=50, maxprice: int=5000, itemtitle: str='paper', categoryid: str='alb-201346210', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Alibaba products by title, price, image, orders volume."
    
    """
    url = f"https://otapi-alibaba.p.rapidapi.com/BatchSearchItemsFrame"
    querystring = {'framePosition': frameposition, 'frameSize': framesize, 'language': language, }
    if imageurl:
        querystring['ImageUrl'] = imageurl
    if minprice:
        querystring['MinPrice'] = minprice
    if orderby:
        querystring['OrderBy'] = orderby
    if minvolume:
        querystring['MinVolume'] = minvolume
    if maxprice:
        querystring['MaxPrice'] = maxprice
    if itemtitle:
        querystring['ItemTitle'] = itemtitle
    if categoryid:
        querystring['CategoryId'] = categoryid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-alibaba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_review(language: str, frameposition: int, itemid: str, framesize: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviews for Alibaba products"
    
    """
    url = f"https://otapi-alibaba.p.rapidapi.com/SearchItemReviews"
    querystring = {'language': language, 'framePosition': frameposition, 'ItemId': itemid, 'frameSize': framesize, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-alibaba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_description(itemid: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Alibaba product description"
    
    """
    url = f"https://otapi-alibaba.p.rapidapi.com/GetItemDescription"
    querystring = {'itemId': itemid, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-alibaba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item(language: str, itemid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full data from Alibaba product by Item Id"
    
    """
    url = f"https://otapi-alibaba.p.rapidapi.com/BatchGetItemFullInfo"
    querystring = {'language': language, 'itemId': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-alibaba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vendor(vendorid: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Alibaba seller information"
    
    """
    url = f"https://otapi-alibaba.p.rapidapi.com/GetVendorInfo"
    querystring = {'vendorId': vendorid, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-alibaba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_catalog(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full catalog for Alibaba"
    
    """
    url = f"https://otapi-alibaba.p.rapidapi.com/GetBriefCatalog"
    querystring = {'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-alibaba.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

