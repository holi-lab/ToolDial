import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def feeds_list(start: int, limit: int, tag: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List feeds by category"
    start: The offset of items to be ignored in response for paging
        limit: Number of items returned per response
        tag: The value of browse-categories/display/tag returned in categories/list API
        
    """
    url = f"https://yummly2.p.rapidapi.com/feeds/list"
    querystring = {'start': start, 'limit': limit, }
    if tag:
        querystring['tag'] = tag
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yummly2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_list(offset: int, globalid: str, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List reviews, feedback from other users"
    offset: The offset of items to be ignored in response for paging
        globalid: The value of globalId field returned in feeds/list and feeds/search API
        limit: Number of items returned per response
        
    """
    url = f"https://yummly2.p.rapidapi.com/reviews/list"
    querystring = {'offset': offset, 'globalId': globalid, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yummly2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_list(offset: int, globalid: str, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List reviews, feedback from other users"
    offset: The offset of items to be ignored in response for paging
        globalid: The value of globalId field returned in feeds/list and feeds/search API
        limit: Number of items returned per response
        
    """
    url = f"https://yummly2.p.rapidapi.com/reviews/list"
    querystring = {'offset': offset, 'globalId': globalid, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yummly2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all categories of recipes"
    
    """
    url = f"https://yummly2.p.rapidapi.com/categories/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yummly2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def categories_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all categories of recipes"
    
    """
    url = f"https://yummly2.p.rapidapi.com/categories/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yummly2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feeds_list_similarities(is_id: str, limit: int, start: int, apifeedtype: str='moreFrom', authorid: str='Yummly', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List similar recipes by name and author"
    id: The value of id field returned in feeds/list and feeds/search API
        limit: Number of items returned per response
        start: The offset of items to be ignored in response for paging
        apifeedtype: The value of apiFeedType field returned in feeds/list and feeds/search API
        authorid: The value of authorId field returned in feeds/list and feeds/search API
        
    """
    url = f"https://yummly2.p.rapidapi.com/feeds/list-similarities"
    querystring = {'id': is_id, 'limit': limit, 'start': start, }
    if apifeedtype:
        querystring['apiFeedType'] = apifeedtype
    if authorid:
        querystring['authorId'] = authorid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yummly2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feeds_list_similarities(is_id: str, limit: int, start: int, apifeedtype: str='moreFrom', authorid: str='Yummly', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List similar recipes by name and author"
    id: The value of id field returned in feeds/list and feeds/search API
        limit: Number of items returned per response
        start: The offset of items to be ignored in response for paging
        apifeedtype: The value of apiFeedType field returned in feeds/list and feeds/search API
        authorid: The value of authorId field returned in feeds/list and feeds/search API
        
    """
    url = f"https://yummly2.p.rapidapi.com/feeds/list-similarities"
    querystring = {'id': is_id, 'limit': limit, 'start': start, }
    if apifeedtype:
        querystring['apiFeedType'] = apifeedtype
    if authorid:
        querystring['authorId'] = authorid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yummly2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feeds_search(start: int, maxresult: int, fibtgmax: int=None, camax: int=None, cholemax: int=None, allowedattribute: str='diet-lacto-vegetarian,diet-low-fodmap', sweetmax: int=None, kmax: int=None, namax: str=None, q: str='chicken soup', enerc_kcalmax: int=None, femax: int=None, fat_kcalmax: int=1000, maxtotaltimeinseconds: int=7200, piquantmax: int=None, vita_iumax: int=None, vitcmax: int=None, meatymax: int=None, fasatmax: int=None, sweetmin: int=None, piquantmin: int=None, fatmax: int=None, sourmin: int=None, meatymin: int=None, sourmax: int=None, chocdfmax: int=None, saltymin: int=None, sugarmax: int=None, procntmax: int=None, saltymax: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for recipes by name and option filters"
    start: The offset of items to be ignored in response for paging
        maxresult: Number of items returned per response
        cholemax: Range from 0 to 1
        allowedattribute: The value of id field returned in tags/list (separated by comma for multiple value)
        sweetmax: How much sweet the meal is, such as 0.8 (0 to 1)
        q: Food name or ingredient
        enerc_kcalmax: Range from 0 to 1000
        fat_kcalmax: Range from 0 to 1000
        maxtotaltimeinseconds: the time (in seconds) it takes to complete the dish
        piquantmax: How much piquant the meal is, such as 0.8 (0 to 1)
        meatymax: How much meaty the meal is, such as 0.8 (0 to 1)
        fasatmax: Range from 0 to 50
        sweetmin: How much sweet the meal is, such as 0.8 (0 to 1)
        piquantmin: How much piquant the meal is, such as 0.8 (0 to 1)
        sourmin: How much sour the meal is, such as 0.8 (0 to 1)
        meatymin: How much meaty the meal is, such as 0.2 (0 to 1)
        sourmax: How much sour the meal is, such as 0.8 (0 to 1)
        chocdfmax: Range from 0 to 100
        saltymin: How much salty the meal is, such as 0.8 (0 to 1)
        saltymax: How much salty the meal is, such as 0.8 (0 to 1)
        
    """
    url = f"https://yummly2.p.rapidapi.com/feeds/search"
    querystring = {'start': start, 'maxResult': maxresult, }
    if fibtgmax:
        querystring['FIBTGMax'] = fibtgmax
    if camax:
        querystring['CAMax'] = camax
    if cholemax:
        querystring['CHOLEMax'] = cholemax
    if allowedattribute:
        querystring['allowedAttribute'] = allowedattribute
    if sweetmax:
        querystring['sweetMax'] = sweetmax
    if kmax:
        querystring['KMax'] = kmax
    if namax:
        querystring['NAMax'] = namax
    if q:
        querystring['q'] = q
    if enerc_kcalmax:
        querystring['ENERC_KCALMax'] = enerc_kcalmax
    if femax:
        querystring['FEMax'] = femax
    if fat_kcalmax:
        querystring['FAT_KCALMax'] = fat_kcalmax
    if maxtotaltimeinseconds:
        querystring['maxTotalTimeInSeconds'] = maxtotaltimeinseconds
    if piquantmax:
        querystring['piquantMax'] = piquantmax
    if vita_iumax:
        querystring['VITA_IUMax'] = vita_iumax
    if vitcmax:
        querystring['VITCMax'] = vitcmax
    if meatymax:
        querystring['meatyMax'] = meatymax
    if fasatmax:
        querystring['FASATMax'] = fasatmax
    if sweetmin:
        querystring['sweetMin'] = sweetmin
    if piquantmin:
        querystring['piquantMin'] = piquantmin
    if fatmax:
        querystring['FATMax'] = fatmax
    if sourmin:
        querystring['sourMin'] = sourmin
    if meatymin:
        querystring['meatyMin'] = meatymin
    if sourmax:
        querystring['sourMax'] = sourmax
    if chocdfmax:
        querystring['CHOCDFMax'] = chocdfmax
    if saltymin:
        querystring['saltyMin'] = saltymin
    if sugarmax:
        querystring['SUGARMax'] = sugarmax
    if procntmax:
        querystring['PROCNTMax'] = procntmax
    if saltymax:
        querystring['saltyMax'] = saltymax
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yummly2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feeds_search(start: int, maxresult: int, fibtgmax: int=None, camax: int=None, cholemax: int=None, allowedattribute: str='diet-lacto-vegetarian,diet-low-fodmap', sweetmax: int=None, kmax: int=None, namax: str=None, q: str='chicken soup', enerc_kcalmax: int=None, femax: int=None, fat_kcalmax: int=1000, maxtotaltimeinseconds: int=7200, piquantmax: int=None, vita_iumax: int=None, vitcmax: int=None, meatymax: int=None, fasatmax: int=None, sweetmin: int=None, piquantmin: int=None, fatmax: int=None, sourmin: int=None, meatymin: int=None, sourmax: int=None, chocdfmax: int=None, saltymin: int=None, sugarmax: int=None, procntmax: int=None, saltymax: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for recipes by name and option filters"
    start: The offset of items to be ignored in response for paging
        maxresult: Number of items returned per response
        cholemax: Range from 0 to 1
        allowedattribute: The value of id field returned in tags/list (separated by comma for multiple value)
        sweetmax: How much sweet the meal is, such as 0.8 (0 to 1)
        q: Food name or ingredient
        enerc_kcalmax: Range from 0 to 1000
        fat_kcalmax: Range from 0 to 1000
        maxtotaltimeinseconds: the time (in seconds) it takes to complete the dish
        piquantmax: How much piquant the meal is, such as 0.8 (0 to 1)
        meatymax: How much meaty the meal is, such as 0.8 (0 to 1)
        fasatmax: Range from 0 to 50
        sweetmin: How much sweet the meal is, such as 0.8 (0 to 1)
        piquantmin: How much piquant the meal is, such as 0.8 (0 to 1)
        sourmin: How much sour the meal is, such as 0.8 (0 to 1)
        meatymin: How much meaty the meal is, such as 0.2 (0 to 1)
        sourmax: How much sour the meal is, such as 0.8 (0 to 1)
        chocdfmax: Range from 0 to 100
        saltymin: How much salty the meal is, such as 0.8 (0 to 1)
        saltymax: How much salty the meal is, such as 0.8 (0 to 1)
        
    """
    url = f"https://yummly2.p.rapidapi.com/feeds/search"
    querystring = {'start': start, 'maxResult': maxresult, }
    if fibtgmax:
        querystring['FIBTGMax'] = fibtgmax
    if camax:
        querystring['CAMax'] = camax
    if cholemax:
        querystring['CHOLEMax'] = cholemax
    if allowedattribute:
        querystring['allowedAttribute'] = allowedattribute
    if sweetmax:
        querystring['sweetMax'] = sweetmax
    if kmax:
        querystring['KMax'] = kmax
    if namax:
        querystring['NAMax'] = namax
    if q:
        querystring['q'] = q
    if enerc_kcalmax:
        querystring['ENERC_KCALMax'] = enerc_kcalmax
    if femax:
        querystring['FEMax'] = femax
    if fat_kcalmax:
        querystring['FAT_KCALMax'] = fat_kcalmax
    if maxtotaltimeinseconds:
        querystring['maxTotalTimeInSeconds'] = maxtotaltimeinseconds
    if piquantmax:
        querystring['piquantMax'] = piquantmax
    if vita_iumax:
        querystring['VITA_IUMax'] = vita_iumax
    if vitcmax:
        querystring['VITCMax'] = vitcmax
    if meatymax:
        querystring['meatyMax'] = meatymax
    if fasatmax:
        querystring['FASATMax'] = fasatmax
    if sweetmin:
        querystring['sweetMin'] = sweetmin
    if piquantmin:
        querystring['piquantMin'] = piquantmin
    if fatmax:
        querystring['FATMax'] = fatmax
    if sourmin:
        querystring['sourMin'] = sourmin
    if meatymin:
        querystring['meatyMin'] = meatymin
    if sourmax:
        querystring['sourMax'] = sourmax
    if chocdfmax:
        querystring['CHOCDFMax'] = chocdfmax
    if saltymin:
        querystring['saltyMin'] = saltymin
    if sugarmax:
        querystring['SUGARMax'] = sugarmax
    if procntmax:
        querystring['PROCNTMax'] = procntmax
    if saltymax:
        querystring['saltyMax'] = saltymax
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yummly2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feeds_auto_complete(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto complete suggestions by name or ingredients, etc..."
    q: Food name or ingredient
        
    """
    url = f"https://yummly2.p.rapidapi.com/feeds/auto-complete"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yummly2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feeds_auto_complete(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto complete suggestions by name or ingredients, etc..."
    q: Food name or ingredient
        
    """
    url = f"https://yummly2.p.rapidapi.com/feeds/auto-complete"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yummly2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tags_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all tags in which recipes are group together"
    
    """
    url = f"https://yummly2.p.rapidapi.com/tags/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yummly2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tags_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all tags in which recipes are group together"
    
    """
    url = f"https://yummly2.p.rapidapi.com/tags/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yummly2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

