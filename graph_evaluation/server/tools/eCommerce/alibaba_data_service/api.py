import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def itemreviewinfo(offer_id: str, page_size: str='20', page_num: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search item review information"
    offer_id: item id
        page_size: records per page，max value 20
        page_num: page number
        
    """
    url = f"https://alibaba-data-service.p.rapidapi.com/Offer/OfferRateInfo.ashx"
    querystring = {'offer_id': offer_id, }
    if page_size:
        querystring['page_size'] = page_size
    if page_num:
        querystring['page_num'] = page_num
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alibaba-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def samestyleitemsearch(offer_id: str, page_num: str='1', page_size: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "same style item search by item id"
    offer_id: item id
        page_num: page number
        page_size: records per page，max value 20
        
    """
    url = f"https://alibaba-data-service.p.rapidapi.com/Search/SearchSameOffers.ashx"
    querystring = {'offer_id': offer_id, }
    if page_num:
        querystring['page_num'] = page_num
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alibaba-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def similarstyleitemsearch(offer_id: str, page_size: str='20', page_num: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "similar style item search by item id"
    page_size: records per page，max value 20
        page_num: page number
        
    """
    url = f"https://alibaba-data-service.p.rapidapi.com/Search/SearchSimilarOffers.ashx"
    querystring = {'offer_id': offer_id, }
    if page_size:
        querystring['page_size'] = page_size
    if page_num:
        querystring['page_num'] = page_num
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alibaba-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def itemdescdetail(offer_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "item detailed description"
    offer_id: item id
        
    """
    url = f"https://alibaba-data-service.p.rapidapi.com/Offer/OfferDescInfo.ashx"
    querystring = {'offer_id': offer_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alibaba-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shopdsrinfo(member_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "detail seller rating"
    member_id: seller member id
        
    """
    url = f"https://alibaba-data-service.p.rapidapi.com/Shop/ShopDsrInfoQuery.ashx"
    querystring = {'member_id': member_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alibaba-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shopauthinfo(member_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "shop auth information"
    member_id: seller member id
        
    """
    url = f"https://alibaba-data-service.p.rapidapi.com/Shop/ShopAuthInfoQuery.ashx"
    querystring = {'member_id': member_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alibaba-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shopcompanyinfo(member_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "seller company information"
    
    """
    url = f"https://alibaba-data-service.p.rapidapi.com/Shop/ShopCompanyInfoQuery.ashx"
    querystring = {'member_id': member_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alibaba-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shopcategory(member_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "shop category(NOT ITEM CATEGORY)"
    
    """
    url = f"https://alibaba-data-service.p.rapidapi.com/Shop/ShopCategoryInfoQuery.ashx"
    querystring = {'member_id': member_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alibaba-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def itemcategory(parent_id: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "item category information"
    parent_id: category id
        
    """
    url = f"https://alibaba-data-service.p.rapidapi.com/Category/CategoryBatchGet.ashx"
    querystring = {}
    if parent_id:
        querystring['parent_id'] = parent_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alibaba-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotsearchkeywords(result_count: str='50', keyword: str=None, category_id: str='80', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "hot item search keywords"
    result_count: return keywords count
        keyword: keyword
        category_id: item category id
        
    """
    url = f"https://alibaba-data-service.p.rapidapi.com/Category/CategoryHotKeywordsGet.ashx"
    querystring = {}
    if result_count:
        querystring['result_count'] = result_count
    if keyword:
        querystring['keyword'] = keyword
    if category_id:
        querystring['category_id'] = category_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alibaba-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def iteminfodetail(offer_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "item detailed infomation"
    offer_id: item id
        
    """
    url = f"https://alibaba-data-service.p.rapidapi.com/Offer/OfferDetailInfo.ashx"
    querystring = {'offer_id': offer_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alibaba-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

