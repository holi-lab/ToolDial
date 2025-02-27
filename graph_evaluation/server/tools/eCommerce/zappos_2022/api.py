import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_brandlist_by_siteid_zappos_1_6pm_2(siteid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET BrandList by siteId (zappos =1, 6pm=2)"
    
    """
    url = f"https://zappos-2022.p.rapidapi.com/brandList"
    querystring = {'siteId': siteid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zappos-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_likecounts_by_itemids_styleids(siteid: int, itemids: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET likeCounts by itemIds(styleIds)"
    
    """
    url = f"https://zappos-2022.p.rapidapi.com/listItemCounts"
    querystring = {'siteId': siteid, 'itemIds': itemids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zappos-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_relatedproducts_by_productid_siteid_zappos_1_6pm_2(productid: int, siteid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get relatedProducts by siteId & productId"
    
    """
    url = f"https://zappos-2022.p.rapidapi.com/relatedProducts"
    querystring = {'productId': productid, 'siteId': siteid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zappos-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_images_by_productid_siteid_zappos_1_6pm_2(siteid: int, productid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET images by productId & siteId"
    
    """
    url = f"https://zappos-2022.p.rapidapi.com/images"
    querystring = {'siteId': siteid, 'productId': productid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zappos-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stocks_info_by_styleids_siteid_zappos_1_6pm_2(siteid: int, productid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET stocks info by siteId & styleIds"
    
    """
    url = f"https://zappos-2022.p.rapidapi.com/product/style/stock"
    querystring = {'siteId': siteid, 'productId': productid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zappos-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_price_by_styleids_and_siteid_zappos_1_6pm_2(siteid: int, styleids: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET price by styleIds and shopId"
    
    """
    url = f"https://zappos-2022.p.rapidapi.com/decorateReco"
    querystring = {'siteId': siteid, 'styleIds': styleids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zappos-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

