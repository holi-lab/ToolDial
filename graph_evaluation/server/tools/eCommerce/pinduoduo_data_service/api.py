import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def multigoodsbasicinfo(goods_ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "query multiple goods basic information"
    goods_ids: array of goods id
        
    """
    url = f"https://pinduoduo-data-service.p.rapidapi.com/Good/GoodsBasic.ashx"
    querystring = {'goods_ids': goods_ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goodssimpleinfo(goods_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "goods simple information"
    goods_id: goods id
        
    """
    url = f"https://pinduoduo-data-service.p.rapidapi.com/Good/GoodsSimple.ashx"
    querystring = {'goods_id': goods_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goodsdetailinfo(goods_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "goods detail information"
    goods_id: goods id
        
    """
    url = f"https://pinduoduo-data-service.p.rapidapi.com/Good/GoodsDetail.ashx"
    querystring = {'goods_id': goods_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topsellerslist(page_num: str, page_size: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "top sellers goods list"
    page_num: page number
        page_size: records per page，max value 20
        
    """
    url = f"https://pinduoduo-data-service.p.rapidapi.com/Ddk/TopGoodsList.ashx"
    querystring = {'page_num': page_num, 'page_size': page_size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goodssearchinshop(mall_id: str, page_num: str='1', page_size: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search goods in one shop"
    mall_id: shop id
        page_num: page number
        page_size: records per page，max value 20
        
    """
    url = f"https://pinduoduo-data-service.p.rapidapi.com/Search/GoodsSearchShop.ashx"
    querystring = {'mall_id': mall_id, }
    if page_num:
        querystring['page_num'] = page_num
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goodssearchbycategory(cat_id: str, page_num: str='1', sort: str='0', end_price: str=None, page_size: str='20', start_price: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search goods by category"
    cat_id: goods category id
        page_num: page number
        sort: sort
0-default
1-order by rate desc
2-order by sales desc
3-order by price desc
4-order by price asc

        end_price: filter by price, need to be paired with 'start_price'
        page_size: records per page，max value 20 
        start_price: filter by price, need to be paired with 'end_price'
        
    """
    url = f"https://pinduoduo-data-service.p.rapidapi.com/Search/GoodsSearchCat.ashx"
    querystring = {'cat_id': cat_id, }
    if page_num:
        querystring['page_num'] = page_num
    if sort:
        querystring['sort'] = sort
    if end_price:
        querystring['end_price'] = end_price
    if page_size:
        querystring['page_size'] = page_size
    if start_price:
        querystring['start_price'] = start_price
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goodscategory(cat_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "goods category"
    cat_id: goods category id
        
    """
    url = f"https://pinduoduo-data-service.p.rapidapi.com/Good/GoodsCat.ashx"
    querystring = {'cat_id': cat_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goodstag(opt_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "goods tag"
    opt_id: goods tag id
        
    """
    url = f"https://pinduoduo-data-service.p.rapidapi.com/Good/GoodsOpt.ashx"
    querystring = {'opt_id': opt_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def goodssearchbytag(opt_id: str, page_num: str='1', sort: str='0', end_price: str=None, page_size: str='10', start_price: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search goods by tag"
    opt_id: goods tag id
        page_num: page number
        sort: sort
0-default
1-order by rate desc
2-order by sales desc
3-order by price desc
4-order by price asc
        end_price: filter by price, need to be paired with 'start_price'
        page_size: records per page，max value 20
        start_price: filter by price, need to be paired with 'end_price'
        
    """
    url = f"https://pinduoduo-data-service.p.rapidapi.com/Search/GoodsSearchOpt.ashx"
    querystring = {'opt_id': opt_id, }
    if page_num:
        querystring['page_num'] = page_num
    if sort:
        querystring['sort'] = sort
    if end_price:
        querystring['end_price'] = end_price
    if page_size:
        querystring['page_size'] = page_size
    if start_price:
        querystring['start_price'] = start_price
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shopsearch(page_size: str='10', cat_id: str=None, mall_ids: str=None, page_num: str='1', merchant_types: str=None, with_goods: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "shop search"
    page_size: records per page，max value 20
        cat_id: goods category id
        mall_ids: array of shop id, max  count 10
        page_num: page number
        merchant_types: shop type
1-person
2-nterprise
3-flagship store
4-specialty store
5-franchise store
6-normal store
        with_goods: return data include goods list，1-yes 0-no
        
    """
    url = f"https://pinduoduo-data-service.p.rapidapi.com/Ddk/MerchantList.ashx"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if cat_id:
        querystring['cat_id'] = cat_id
    if mall_ids:
        querystring['mall_ids'] = mall_ids
    if page_num:
        querystring['page_num'] = page_num
    if merchant_types:
        querystring['merchant_types'] = merchant_types
    if with_goods:
        querystring['with_goods'] = with_goods
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinduoduo-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

