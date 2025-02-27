import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def screenshots(is_id: str, limit: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**get gameplay screenshots**"
    limit: **limit of 100 !**
        
    """
    url = f"https://games-details.p.rapidapi.com/{is_id}/screenshots/{limit}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "games-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def official_announcement(is_id: str, limit: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**get official announcement news**"
    limit: **limit of 100 !**
        
    """
    url = f"https://games-details.p.rapidapi.com/{is_id}/news/announcements/{limit}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "games-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def allnews(is_id: str, limit: str='30', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**get allnews**"
    limit: **limit of 100 !**
        
    """
    url = f"https://games-details.p.rapidapi.com/{is_id}/news/allnews/{limit}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "games-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trendday_reviews(is_id: str, limit: str='30', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Trendday reviews with limit of 100"
    limit: **limit of 100 !**
        
    """
    url = f"https://games-details.p.rapidapi.com/{is_id}/reviews/trendday/{limit}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "games-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def funny_reviews(is_id: str, limit: str='30', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**get funny reviews**"
    limit: **limit of 100 !**
        
    """
    url = f"https://games-details.p.rapidapi.com/{is_id}/reviews/funny/{limit}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "games-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mostrecent_reviews(is_id: str, limit: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**get mostrecent reviews**"
    limit: **In some cases reviews **officially** is less than the given limit and max limit is100 !**
        
    """
    url = f"https://games-details.p.rapidapi.com/{is_id}/reviews/mostrecent/{limit}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "games-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trendyear_reviews(is_id: str, limit: str='30', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**get trendyear reviews !**"
    limit: **limit of 100 !**
        
    """
    url = f"https://games-details.p.rapidapi.com/{is_id}/reviews/trendyear/{limit}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "games-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trendmonth_reviews(is_id: str, limit: str='30', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**get Trendmonth reviews !**"
    limit: **limit of 100 !**
        
    """
    url = f"https://games-details.p.rapidapi.com/{is_id}/reviews/trendmonth/{limit}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "games-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trendweek_reviews(is_id: str, limit: str='30', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "trendweek reviews with limit of 100."
    limit: **limit of 100 !**
        
    """
    url = f"https://games-details.p.rapidapi.com/{is_id}/reviews/trendweek/{limit}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "games-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def toprated_reviews(is_id: str, limit: str='30', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get **TopRated** Reviews with **limit of 100** .It also suggest the review is Recommended or not Recommended in title."
    
    """
    url = f"https://games-details.p.rapidapi.com/{is_id}/reviews/toprated/{limit}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "games-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_game(sugg: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**THIS ENDPOINT WILL RETURN ALL GAMES WHICH RELATED TO  PARAMETER**"
    
    """
    url = f"https://games-details.p.rapidapi.com/search/{sugg}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "games-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def single_game(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**GET FULL DETAILS OF SINGLE GAME BY *ID* WITH SYSTEM REQUIREMENT**"
    
    """
    url = f"https://games-details.p.rapidapi.com/single_game/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "games-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def page_number(page_no: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "**THERE ARE AROUND 5538 PAGES AND SINGLE PAGE CONTAIN 25 GAMES**"
    
    """
    url = f"https://games-details.p.rapidapi.com/page/{page_no}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "games-details.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

