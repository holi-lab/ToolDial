import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def shopinfobasic(shop_id: str=None, vender_id: str='1000101275', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "shop basic information"
    shop_id: shop id
        vender_id: vender id
        
    """
    url = f"https://jd-com-data-service.p.rapidapi.com/Shop/ShopBasicInfoGet.ashx"
    querystring = {}
    if shop_id:
        querystring['shop_id'] = shop_id
    if vender_id:
        querystring['vender_id'] = vender_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jd-com-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def productreviewinfo(product_id: str, score: str='0', only_sku: str='0', sort: str=None, page_num: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search product review information"
    product_id: product id
        score: review type
3-good
2-neutral
1-bad
0-all
        only_sku: only show one sku review info
1-true
0-false
        page_num: page number
        
    """
    url = f"https://jd-com-data-service.p.rapidapi.com/Comment/CommentListGet.ashx"
    querystring = {'product_id': product_id, }
    if score:
        querystring['score'] = score
    if only_sku:
        querystring['only_sku'] = only_sku
    if sort:
        querystring['sort'] = sort
    if page_num:
        querystring['page_num'] = page_num
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jd-com-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def productinfobasic(product_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "product basic infomation"
    product_id: product id
        
    """
    url = f"https://jd-com-data-service.p.rapidapi.com/Product/ProductDetailGet.ashx"
    querystring = {'product_id': product_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jd-com-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def productinfodetail(product_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "product detailed infomation"
    product_id: product id
        
    """
    url = f"https://jd-com-data-service.p.rapidapi.com/Product/ProductSkuViewGet.ashx"
    querystring = {'product_id': product_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jd-com-data-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

