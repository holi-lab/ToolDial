import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def author(media: str=None, name: str=None, size: int=None, page: int=None, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find article's authors"
    media: Comma separated list of medias e.g (bbc,NYtimes)
        name: Comma separated list of categories e.g(Sport,Politic)
        size: Specify the number of articles you want to return (Max is 10)
        
    """
    url = f"https://youpel-newsapi.p.rapidapi.com/authors"
    querystring = {}
    if media:
        querystring['media'] = media
    if name:
        querystring['name'] = name
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youpel-newsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def media(country: str=None, size: int=None, media: str=None, category: str=None, page: int=None, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find article's medias"
    size: Specify the number of articles you want to return (Max is 10)
        media: Comma separated list of medias e.g (bbc,NYtimes)
        category: Comma separated list of categories e.g(Sport,Politic)
        
    """
    url = f"https://youpel-newsapi.p.rapidapi.com/medias"
    querystring = {}
    if country:
        querystring['country'] = country
    if size:
        querystring['size'] = size
    if media:
        querystring['media'] = media
    if category:
        querystring['category'] = category
    if page:
        querystring['page'] = page
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youpel-newsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def category(country: str=None, page: int=None, category: str=None, language: str=None, media: str=None, size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find article's categories"
    category: Comma separated list of categories e.g(Sport,Politic)
        media: Comma separated list of medias e.g (bbc,NYtimes)
        size: Specify the number of articles you want to return (Max is 10)
        
    """
    url = f"https://youpel-newsapi.p.rapidapi.com/categories"
    querystring = {}
    if country:
        querystring['country'] = country
    if page:
        querystring['page'] = page
    if category:
        querystring['category'] = category
    if language:
        querystring['language'] = language
    if media:
        querystring['media'] = media
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youpel-newsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def language(size: int=None, language: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find article's languages"
    size: Specify the number of articles you want to return
        
    """
    url = f"https://youpel-newsapi.p.rapidapi.com/languages"
    querystring = {}
    if size:
        querystring['size'] = size
    if language:
        querystring['language'] = language
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youpel-newsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geo(state: str=None, country: str=None, size: int=None, page: int=None, city: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find article's geographical informations"
    size: Specify the number of articles you want to return
        
    """
    url = f"https://youpel-newsapi.p.rapidapi.com/geo"
    querystring = {}
    if state:
        querystring['state'] = state
    if country:
        querystring['country'] = country
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    if city:
        querystring['city'] = city
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youpel-newsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles(media: str=None, to: str=None, size: int=None, city: str=None, is_from: str=None, page: int=None, q: str=None, country: str=None, language: str=None, category: str=None, author: str=None, state: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news articles relevant for a given query."
    media: Comma separated list of medias e.g (bbc,NYtimes)
        to: to_ accept the following format of date: %Y-%m-%d For example, 2022-06-01
        size: Specify the number of articles you want to return (Max is 10)
        is_from: from_ accept the following format of date: %Y-%m-%d For example, 2022-04-01
        category: Comma separated list of categories e.g(Sport,Politic)
        
    """
    url = f"https://youpel-newsapi.p.rapidapi.com/articles"
    querystring = {}
    if media:
        querystring['media'] = media
    if to:
        querystring['to'] = to
    if size:
        querystring['size'] = size
    if city:
        querystring['city'] = city
    if is_from:
        querystring['from'] = is_from
    if page:
        querystring['page'] = page
    if q:
        querystring['q'] = q
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    if category:
        querystring['category'] = category
    if author:
        querystring['author'] = author
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "youpel-newsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

