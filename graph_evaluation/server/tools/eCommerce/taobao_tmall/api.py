import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_for_items(frameposition: int, language: str, framesize: int, isstock: bool=None, imageurl: str='https://img.alicdn.com/bao/uploaded/i4/2773693278/O1CN01YrlRK91a5MwEi1osq_!!2773693278.jpg_500x500.jpg', maxprice: int=5000, iscomplete: bool=None, minprice: int=100, vendorname: str='莹儿225', orderby: str='UpdatedTime:Desc', itemtitle: str='glasses', minvolume: int=50, categoryid: str='50023674', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search fo Toabao or Tmall products by title, price, image, orders volume."
    
    """
    url = f"https://taobao-tmall1.p.rapidapi.com/BatchSearchItemsFrame"
    querystring = {'framePosition': frameposition, 'language': language, 'frameSize': framesize, }
    if isstock:
        querystring['IsStock'] = isstock
    if imageurl:
        querystring['ImageUrl'] = imageurl
    if maxprice:
        querystring['MaxPrice'] = maxprice
    if iscomplete:
        querystring['IsComplete'] = iscomplete
    if minprice:
        querystring['MinPrice'] = minprice
    if vendorname:
        querystring['VendorName'] = vendorname
    if orderby:
        querystring['OrderBy'] = orderby
    if itemtitle:
        querystring['ItemTitle'] = itemtitle
    if minvolume:
        querystring['MinVolume'] = minvolume
    if categoryid:
        querystring['CategoryId'] = categoryid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_full_info(language: str, itemid: str, targetareacode: str='110103', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full data from Toabao or Tmall product by Item Id"
    
    """
    url = f"https://taobao-tmall1.p.rapidapi.com/BatchGetItemFullInfo"
    querystring = {'language': language, 'itemId': itemid, }
    if targetareacode:
        querystring['TargetAreaCode'] = targetareacode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_deliveryarealist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get  Area codes for delivery request"
    
    """
    url = f"https://taobao-tmall1.p.rapidapi.com/GetProviderAllAreaList"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_review(itemid: str, frameposition: int, framesize: int, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviews for Toabao and Tmall products"
    
    """
    url = f"https://taobao-tmall1.p.rapidapi.com/SearchItemReviews"
    querystring = {'ItemId': itemid, 'framePosition': frameposition, 'frameSize': framesize, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_description(language: str, itemid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Toabao or Tmall product description"
    
    """
    url = f"https://taobao-tmall1.p.rapidapi.com/GetItemDescription"
    querystring = {'language': language, 'itemId': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vendor(vendorid: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Taobao seller information"
    
    """
    url = f"https://taobao-tmall1.p.rapidapi.com/GetVendorInfo"
    querystring = {'vendorId': vendorid, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_catalog(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full catalog for Taobao and Tmall"
    
    """
    url = f"https://taobao-tmall1.p.rapidapi.com/GetBriefCatalog"
    querystring = {'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

