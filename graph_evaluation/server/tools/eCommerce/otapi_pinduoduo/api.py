import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_vendor(vendorid: str, language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Pinduoduo seller information"
    
    """
    url = f"https://otapi-pinduoduo.p.rapidapi.com/GetVendorInfo"
    querystring = {'vendorId': vendorid, 'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-pinduoduo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_catalog(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full catalog for Pinduoduo"
    
    """
    url = f"https://otapi-pinduoduo.p.rapidapi.com/GetBriefCatalog"
    querystring = {'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-pinduoduo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_items(language: str, frameposition: int, framesize: int, imageurl: str=None, orderby: str='Popularity:Desc', maxprice: str='5000', minprice: int=100, categoryid: str='pd-11464', itemtitle: str='glasses', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for Pinduoduo products by title, price, image, orders volume."
    
    """
    url = f"https://otapi-pinduoduo.p.rapidapi.com/BatchSearchItemsFrame"
    querystring = {'language': language, 'framePosition': frameposition, 'frameSize': framesize, }
    if imageurl:
        querystring['ImageUrl'] = imageurl
    if orderby:
        querystring['OrderBy'] = orderby
    if maxprice:
        querystring['MaxPrice'] = maxprice
    if minprice:
        querystring['MinPrice'] = minprice
    if categoryid:
        querystring['CategoryId'] = categoryid
    if itemtitle:
        querystring['ItemTitle'] = itemtitle
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-pinduoduo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item(language: str, itemid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data from Pinduoduo product by Item Id (no SKU data)"
    
    """
    url = f"https://otapi-pinduoduo.p.rapidapi.com/BatchGetItemFullInfo"
    querystring = {'language': language, 'itemId': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-pinduoduo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

