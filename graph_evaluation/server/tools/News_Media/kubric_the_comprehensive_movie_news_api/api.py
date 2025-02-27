import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sources(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns sources used for articles based of current tier."
    
    """
    url = f"https://kubric-the-comprehensive-movie-news-api.p.rapidapi.com/news/sources"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kubric-the-comprehensive-movie-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def advanced_search(q: str, sentiment: str='postive', maxdate: str='05/08/2023', mindate: str='01/01/2023', type: str='Article', offset: int=0, limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The `/news/advanced/search` endpoint allows you to search for movie news articles by applying various filters such as date, source, type, and sentiment. It offers a versatile and customizable way to explore the latest news and updates in the movie industry."
    
    """
    url = f"https://kubric-the-comprehensive-movie-news-api.p.rapidapi.com/news/advanced/search"
    querystring = {'q': q, }
    if sentiment:
        querystring['sentiment'] = sentiment
    if maxdate:
        querystring['maxDate'] = maxdate
    if mindate:
        querystring['minDate'] = mindate
    if type:
        querystring['type'] = type
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kubric-the-comprehensive-movie-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def basic_search(q: str, limit: int=10, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The `/news/basic/search` endpoint allows you to search for movie news articles."
    
    """
    url = f"https://kubric-the-comprehensive-movie-news-api.p.rapidapi.com/news/basic/search"
    querystring = {'q': q, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kubric-the-comprehensive-movie-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The `/trending` endpoint is a powerful feature of the Kubric Movie News API, designed to provide users with the most recent and popular movie news articles. This endpoint returns the top 50 trending movie articles, ensuring users stay up-to-date with the latest and most engaging content from the world of cinema."
    
    """
    url = f"https://kubric-the-comprehensive-movie-news-api.p.rapidapi.com/news/trending"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kubric-the-comprehensive-movie-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news(source: str='Hollywood Reporter', type: str='Article', offset: int=0, sentiment: str='positive', limit: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The `/news` endpoint provides you with the most recent movie news articles, keeping you informed about the latest updates, events, and releases in the film industry. This endpoint offers a quick and easy way to stay up-to-date with the latest happenings in the movie world without any additional filters or search parameters."
    
    """
    url = f"https://kubric-the-comprehensive-movie-news-api.p.rapidapi.com/news"
    querystring = {}
    if source:
        querystring['source'] = source
    if type:
        querystring['type'] = type
    if offset:
        querystring['offset'] = offset
    if sentiment:
        querystring['sentiment'] = sentiment
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kubric-the-comprehensive-movie-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

