import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def global_achievement_percentages_for_app(appid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of all achievements and their percentages success rate"
    
    """
    url = f"https://steam2.p.rapidapi.com/globalAchievementPercentagesForApp/{appid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_news(limit: int, appid: int, contentlength: int=300, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get last n news for App"
    contentlength: Default 300
        
    """
    url = f"https://steam2.p.rapidapi.com/newsForApp/{appid}/limit/{limit}/{contentlength}"
    querystring = {}
    if contentlength:
        querystring['contentLength'] = contentlength
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_reviews(appid: str, limit: int, cursor: str='*', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get last n reviews of an app"
    limit: Max limit 200
        cursor: Reviews are returned in batches max 200 (instead of 20 as in original API), so pass \"*\" for the first set, then the value of \"cursor\" that was returned in the response for the next set, etc. Note that cursor values may contain characters that need to be URLEncoded for use in the query string. As default is \"*\"
        
    """
    url = f"https://steam2.p.rapidapi.com/appReviews/{appid}/limit/{limit}/{cursor}"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_detail(appid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed info of the app"
    
    """
    url = f"https://steam2.p.rapidapi.com/appDetail/{appid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(term: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search in steam shop"
    page: Default 1
        
    """
    url = f"https://steam2.p.rapidapi.com/search/{term}/page/{page}"
    querystring = {}
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

