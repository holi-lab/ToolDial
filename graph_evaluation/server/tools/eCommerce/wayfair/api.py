import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def products_get_common_info(sku: str, optioncombinationoptions: str=None, wfproductoptions: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get common info services of specific product"
    sku: The value of sku fields returned in …/products/list or …/products/search endpoint.
        optioncombinationoptions: The value of id fields under relevantAttributes JSON object returned right in this endpoint, pass this parameter multiple time for multiple options. Ex : …&wfProductOptions=1234567890&wfProductOptions=special_offers&…
        wfproductoptions: The value of id fields under relevantAttributes JSON object returned right in this endpoint, pass this parameter multiple time for multiple options. Ex : …&wfProductOptions=1234567890&wfProductOptions=special_offers&…
        
    """
    url = f"https://wayfair.p.rapidapi.com/products/get-common-info"
    querystring = {'sku': sku, }
    if optioncombinationoptions:
        querystring['optionCombinationOptions'] = optioncombinationoptions
    if wfproductoptions:
        querystring['wfProductOptions'] = wfproductoptions
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wayfair.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_get_home_services(sku: str, productoptions: str=None, postalcode: str='67346', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of specific product"
    sku: The value of sku fields returned in …/products/list or …/products/search endpoint.
        productoptions: The value of id fields under relevantAttributes JSON object returned right in this endpoint, pass this parameter multiple time for multiple options. Ex : …&wfProductOptions=1234567890&wfProductOptions=special_offers&…
        postalcode: The postal code
        
    """
    url = f"https://wayfair.p.rapidapi.com/products/get-home-services"
    querystring = {'sku': sku, }
    if productoptions:
        querystring['productOptions'] = productoptions
    if postalcode:
        querystring['postalCode'] = postalcode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wayfair.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_get_similar_items(sku: str, optionids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get similar items of specific product"
    sku: The value of sku fields returned in …/products/list or …/products/search endpoint.
        optionids: The value of id fields under relevantAttributes JSON object returned right in this endpoint, pass this parameter multiple time for multiple options. Ex : …&wfProductOptions=1234567890&wfProductOptions=special_offers&…
        
    """
    url = f"https://wayfair.p.rapidapi.com/products/get-similar-items"
    querystring = {'sku': sku, }
    if optionids:
        querystring['optionIds'] = optionids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wayfair.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_get_warranty_services(sku: str, selectedoptionids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get warranty services of specific product"
    sku: The value of sku fields returned in …/products/list or …/products/search endpoint.
        selectedoptionids: The value of id fields under relevantAttributes JSON object returned right in this endpoint, pass this parameter multiple time for multiple options. Ex : …&wfProductOptions=1234567890&wfProductOptions=special_offers&…


        
    """
    url = f"https://wayfair.p.rapidapi.com/products/get-warranty-services"
    querystring = {'sku': sku, }
    if selectedoptionids:
        querystring['selectedOptionIds'] = selectedoptionids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wayfair.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_get_pricing(sku: str, options: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get pricing of specific product"
    sku: The value of sku fields returned in …/products/list or …/products/search endpoint.
        options: The value of id fields under relevantAttributes JSON object returned right in this endpoint, pass this parameter multiple time for multiple options. Ex : …&wfProductOptions=1234567890&wfProductOptions=special_offers&…
        
    """
    url = f"https://wayfair.p.rapidapi.com/products/get-pricing"
    querystring = {'sku': sku, }
    if options:
        querystring['options'] = options
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wayfair.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_detail_deprecated(sku: str, wfproductoptions: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of specific product"
    sku: The value of sku fields returned in .../products/list or .../products/search endpoint.
        wfproductoptions: The value of id fields under relevantAttributes JSON object returned right in this endpoint, pass this parameter multiple time for multiple options. Ex : ...&wfProductOptions=1234567890&wfProductOptions=special_offers&...
        
    """
    url = f"https://wayfair.p.rapidapi.com/products/detail"
    querystring = {'sku': sku, }
    if wfproductoptions:
        querystring['wfProductOptions'] = wfproductoptions
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wayfair.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_get_images(sku: str, selectedoptionids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get images of specific product"
    sku: The value of sku fields returned in …/products/list or …/products/search endpoint.
        selectedoptionids: The value of id fields under relevantAttributes JSON object returned right in this endpoint, pass this parameter multiple time for multiple options. Ex : …&wfProductOptions=1234567890&wfProductOptions=special_offers&…
        
    """
    url = f"https://wayfair.p.rapidapi.com/products/get-images"
    querystring = {'sku': sku, }
    if selectedoptionids:
        querystring['selectedOptionIds'] = selectedoptionids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wayfair.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_get_financing_offers(sku: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get financing offers of specific product"
    sku: The value of sku fields returned in …/products/list or …/products/search endpoint.
        
    """
    url = f"https://wayfair.p.rapidapi.com/products/get-financing-offers"
    querystring = {'sku': sku, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wayfair.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_v2_detail_deprecating(sku: str, wfproductoptions: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of specific product"
    sku: The value of sku fields returned in …/products/list or …/products/search endpoint.
        wfproductoptions: The value of id fields under relevantAttributes JSON object returned right in this endpoint, pass this parameter multiple time for multiple options. Ex : …&wfProductOptions=1234567890&wfProductOptions=special_offers&…
        
    """
    url = f"https://wayfair.p.rapidapi.com/products/v2/detail"
    querystring = {'sku': sku, }
    if wfproductoptions:
        querystring['wfProductOptions'] = wfproductoptions
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wayfair.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_list(sku: str, page: int=1, star: str=None, sort_order: str='RELEVANCE', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List reviews relating to specific product"
    sku: The value of sku fields returned in .../products/list or .../products/search endpoint.
        page: For paging purpose
        star: Leave empty or  1 to 5
        sort_order: One of the following : RELEVANCE|HELPFUL|DATE&#95;ASCENDING|DATE&#95;DESCENDING|IMAGE|RATING&#95;DESCENDING|RATING&#95;ASCENDING
        
    """
    url = f"https://wayfair.p.rapidapi.com/reviews/list"
    querystring = {'sku': sku, }
    if page:
        querystring['page'] = page
    if star:
        querystring['star'] = star
    if sort_order:
        querystring['sort_order'] = sort_order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wayfair.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_list(caid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List categories and their recursive children categories if available"
    caid: The value of categoryId fields returned right in this endpoint. The default root category is 214970.
        
    """
    url = f"https://wayfair.p.rapidapi.com/categories/list"
    querystring = {'caid': caid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wayfair.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto suggestions by term or phrase"
    query: Any term or phrase that you are familiar with
        
    """
    url = f"https://wayfair.p.rapidapi.com/auto-complete"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wayfair.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_list(categoryid: int, currentzipcode: str=None, page: int=1, itemsperpage: int=48, sortid: int=None, filterstringunencoded: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List products with filters and options"
    categoryid: The value of categoryId fields returned in .../categories/list endpoint
        currentzipcode: The postal code to get near available products.
        page: For paging purpose
        itemsperpage: For paging purpose
        sortid: Check availableSorts field returned right in this endpoint for suitable sortId
        filterstringunencoded: The value of filterStringUnencoded fields returned right in this endpoint to filter products, pass this parameter multiple times for multiple filters. Ex : ...&filterStringUnencoded=a1234567890~2147483646&filterStringUnencoded=at&#95;style~Tiffany&...
        
    """
    url = f"https://wayfair.p.rapidapi.com/products/list"
    querystring = {'categoryId': categoryid, }
    if currentzipcode:
        querystring['currentZipCode'] = currentzipcode
    if page:
        querystring['page'] = page
    if itemsperpage:
        querystring['itemsPerPage'] = itemsperpage
    if sortid:
        querystring['sortId'] = sortid
    if filterstringunencoded:
        querystring['filterStringUnencoded'] = filterstringunencoded
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wayfair.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_search(keyword: str, filters: str=None, curpage: int=1, itemsperpage: int=48, sortby: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for products by term or phrase"
    keyword: Any term or phrase to look for relating products
        filters: The value of filter&#95;string&#95;unencoded fields returned right in this endpoint to filter products, pass this parameter multiple times for multiple filters. Ex : ...&filters=colorList~White&filters=masterClID~180&...
        curpage: For paging purpose
        itemsperpage: For paging purpose
        sortby: The value of sort_value fields returned right in this endpoint. Default is 0
        
    """
    url = f"https://wayfair.p.rapidapi.com/products/search"
    querystring = {'keyword': keyword, }
    if filters:
        querystring['filters'] = filters
    if curpage:
        querystring['curpage'] = curpage
    if itemsperpage:
        querystring['itemsperpage'] = itemsperpage
    if sortby:
        querystring['sortby'] = sortby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wayfair.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

