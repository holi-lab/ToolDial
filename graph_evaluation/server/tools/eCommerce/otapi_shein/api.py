import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_for_items(frameposition: int, language: str, framesize: int=50, maxprice: int=100, iscomplete: bool=None, imageurl: str='https://img.ltwebstatic.com/images3_pi/2022/09/21/1663727047984da87141063280ec8f3442bc1d9a22.webp', categoryid: str='3195', orderby: str='UpdatedTime:Desc', minprice: int=1, itemtitle: str='tshort', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search fo Shein products by title, price, image, orders volume."
    
    """
    url = f"https://otapi-shein.p.rapidapi.com/BatchSearchItemsFrame"
    querystring = {'framePosition': frameposition, 'language': language, }
    if framesize:
        querystring['frameSize'] = framesize
    if maxprice:
        querystring['MaxPrice'] = maxprice
    if iscomplete:
        querystring['IsComplete'] = iscomplete
    if imageurl:
        querystring['ImageUrl'] = imageurl
    if categoryid:
        querystring['CategoryId'] = categoryid
    if orderby:
        querystring['OrderBy'] = orderby
    if minprice:
        querystring['MinPrice'] = minprice
    if itemtitle:
        querystring['ItemTitle'] = itemtitle
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_review(frameposition: int, framesize: int, language: str, itemid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviews for Shein  products"
    
    """
    url = f"https://otapi-shein.p.rapidapi.com/SearchItemReviews"
    querystring = {'framePosition': frameposition, 'frameSize': framesize, 'language': language, 'ItemId': itemid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_full_info(language: str, itemid: str='sh-17464632', targetareacode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full data from Shein product by Item Id"
    
    """
    url = f"https://otapi-shein.p.rapidapi.com/BatchGetItemFullInfo"
    querystring = {'language': language, }
    if itemid:
        querystring['itemId'] = itemid
    if targetareacode:
        querystring['TargetAreaCode'] = targetareacode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_catalog(language: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full catalog for Shein global"
    
    """
    url = f"https://otapi-shein.p.rapidapi.com/GetBriefCatalog"
    querystring = {'language': language, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "otapi-shein.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

