import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_news_from_the_whole_world(search: str='President', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will get you news from all over the world."
    
    """
    url = f"https://all-in-one-news-live.p.rapidapi.com/news/world"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-news-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_news_for_the_automobile(search: str='car', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will get you news from all of the world for the automobile industory."
    
    """
    url = f"https://all-in-one-news-live.p.rapidapi.com/news/auto"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-news-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_news_for_the_life_style(search: str='actress', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will get you news from all of the world for the lifestyle."
    
    """
    url = f"https://all-in-one-news-live.p.rapidapi.com/news/lifeAndStyle"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-news-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_news_for_the_entertainment(search: str='movie', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will get you news from all of the world for the entertainment industory."
    
    """
    url = f"https://all-in-one-news-live.p.rapidapi.com/news/entertainment"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-news-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_news_for_the_education(search: str='book', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will get you news from all of the world for the education."
    
    """
    url = f"https://all-in-one-news-live.p.rapidapi.com/news/education"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-news-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_news_for_the_technology_and_gadget(search: str='phone', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will get you news from all of the world for the technology and gadget around."
    
    """
    url = f"https://all-in-one-news-live.p.rapidapi.com/news/tech"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-news-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_news_for_the_environment(search: str='tree', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will get you news from all of the world for environment."
    
    """
    url = f"https://all-in-one-news-live.p.rapidapi.com/news/env"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-news-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_news_for_the_science(search: str='science', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will get you news from all of the world for science."
    
    """
    url = f"https://all-in-one-news-live.p.rapidapi.com/news/science"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-news-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_news_for_the_sports(search: str='bat', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will get you news from all of the world for all the sports combined."
    
    """
    url = f"https://all-in-one-news-live.p.rapidapi.com/news/sports"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-news-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_news_for_the_cricket_game(search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will get you news from all of the world for cricket."
    
    """
    url = f"https://all-in-one-news-live.p.rapidapi.com/news/cricket"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-news-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_news_from_the_whole_usa(search: str='live', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will get you news from all of the USA region."
    
    """
    url = f"https://all-in-one-news-live.p.rapidapi.com/news/US"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-news-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gets_news_from_india_only(search: str='live', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will get you news from india region only."
    
    """
    url = f"https://all-in-one-news-live.p.rapidapi.com/news/india"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-news-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_news(search: str='live', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return the latest news from around the world."
    
    """
    url = f"https://all-in-one-news-live.p.rapidapi.com/news"
    querystring = {}
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-in-one-news-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

