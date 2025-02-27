import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def itemdescdetail(num_iid: str, type: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "item detailed description"
    num_iid: item id
        type: 1-PC site 
2-Mobile Site
        
    """
    url = f"https://taobao-tmall-data-service.p.rapidapi.com/Item/MobileWDetailGetItemDesc.ashx"
    querystring = {'num_iid': num_iid, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def itemdescdetail(num_iid: str, type: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "item detailed description"
    num_iid: item id
        type: 1-PC site 
2-Mobile Site
        
    """
    url = f"https://taobao-tmall-data-service.p.rapidapi.com/Item/MobileWDetailGetItemDesc.ashx"
    querystring = {'num_iid': num_iid, }
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def iteminfodetail2(num_iid: str, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "item detailed infomation (method 2)"
    num_iid: item id
        fields: fields of item objects required to be returned, such as title, price and descmodules. Optional value: all the fields in item struct \"Item\" can be returned, which are separated by \",\". Newly added returned fields: itemweight (item weight in digital format, including decimals), itemsize (item volume in digital format, including decimals), changeprop (data of basic colors of item).
        
    """
    url = f"https://taobao-tmall-data-service.p.rapidapi.com/Item/ItemGet.ashx"
    querystring = {'num_iid': num_iid, }
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def iteminfodetail2(num_iid: str, fields: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "item detailed infomation (method 2)"
    num_iid: item id
        fields: fields of item objects required to be returned, such as title, price and descmodules. Optional value: all the fields in item struct \"Item\" can be returned, which are separated by \",\". Newly added returned fields: itemweight (item weight in digital format, including decimals), itemsize (item volume in digital format, including decimals), changeprop (data of basic colors of item).
        
    """
    url = f"https://taobao-tmall-data-service.p.rapidapi.com/Item/ItemGet.ashx"
    querystring = {'num_iid': num_iid, }
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def iteminfodetail1(num_iid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "item detailed infomation (method 1)"
    num_iid: item id
        
    """
    url = f"https://taobao-tmall-data-service.p.rapidapi.com/Item/MobileDetailGetDetail.ashx"
    querystring = {'num_iid': num_iid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shopcategory(shop_id: str=None, seller_id: str='713464357', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "shop category(NOT ITEM CATEGORY)"
    shop_id: shop id
        seller_id: seller id
        
    """
    url = f"https://taobao-tmall-data-service.p.rapidapi.com/Shop/MobileShopCatNeoGet.ashx"
    querystring = {}
    if shop_id:
        querystring['shop_id'] = shop_id
    if seller_id:
        querystring['seller_id'] = seller_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shopinfodetail1(nick: str=None, shop_id: str='67095450', seller_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "shop detailed infomation (method 1)"
    nick: nick
        shop_id: shop id
        seller_id: seller id
        
    """
    url = f"https://taobao-tmall-data-service.p.rapidapi.com/Shop/MobileShopGetWapShopInfo.ashx"
    querystring = {}
    if nick:
        querystring['nick'] = nick
    if shop_id:
        querystring['shop_id'] = shop_id
    if seller_id:
        querystring['seller_id'] = seller_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shopinfodetail2(seller_id: str=None, shop_id: str='67095450', nick: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "shop detailed infomation (method 2)"
    seller_id: seller id
        shop_id: shop id
        nick: nick
        
    """
    url = f"https://taobao-tmall-data-service.p.rapidapi.com/Shop/MobileGebShopInfoQueryShopInfo.ashx"
    querystring = {}
    if seller_id:
        querystring['seller_id'] = seller_id
    if shop_id:
        querystring['shop_id'] = shop_id
    if nick:
        querystring['nick'] = nick
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shopinfodetail3(shop_id: str='67095450', seller_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "shop detailed infomation (method 3)"
    shop_id: shop id
        seller_id: seller id
        
    """
    url = f"https://taobao-tmall-data-service.p.rapidapi.com/Shop/MobileGebShopAboutQueryShopBrief.ashx"
    querystring = {}
    if shop_id:
        querystring['shop_id'] = shop_id
    if seller_id:
        querystring['seller_id'] = seller_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shopdsrinfo(seller_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "shop detailed infomation about DSR"
    seller_id: seller id
        
    """
    url = f"https://taobao-tmall-data-service.p.rapidapi.com/Shop/WebShopScoreGet.ashx"
    querystring = {'seller_id': seller_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shopcoupons(seller_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get coupons list of shop"
    seller_id: seller id
        
    """
    url = f"https://taobao-tmall-data-service.p.rapidapi.com/Shop/MobileShopQueryBuyerBonus.ashx"
    querystring = {'seller_id': seller_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shopreviewinfo(page_num: str='1', shop_id: str='207013635', rate_type: str='1', seller_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search shop review information"
    page_num: page number
        shop_id: shop id
        rate_type: review type
1-good
0-neutral
-1-bad
        seller_id: seller id
        
    """
    url = f"https://taobao-tmall-data-service.p.rapidapi.com/Rate/MobileShopGetSellerRate.ashx"
    querystring = {}
    if page_num:
        querystring['page_num'] = page_num
    if shop_id:
        querystring['shop_id'] = shop_id
    if rate_type:
        querystring['rate_type'] = rate_type
    if seller_id:
        querystring['seller_id'] = seller_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def itemreviewinfo1(num_iid: str, page_num: str='1', sort: str='2', type: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search item review information (method 1)"
    num_iid: item id
        page_num: page number
        sort: sort
1-default，
2-order by created time desc
        type: review type
all,bad,normal,good,append,picture,video
        
    """
    url = f"https://taobao-tmall-data-service.p.rapidapi.com/Rate/MobileWDetailGetItemRates.ashx"
    querystring = {'num_iid': num_iid, }
    if page_num:
        querystring['page_num'] = page_num
    if sort:
        querystring['sort'] = sort
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def itemreviewinfo2(num_iid: str, page_num: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search item review information (method 2)"
    num_iid: item id
        page_num: page number
        
    """
    url = f"https://taobao-tmall-data-service.p.rapidapi.com/Rate/MobileRateWebDetailRateServiceCoronetGetRateList.ashx"
    querystring = {'num_iid': num_iid, }
    if page_num:
        querystring['page_num'] = page_num
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

