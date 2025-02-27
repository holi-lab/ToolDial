import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def trending(format: str=None, sortby: str=None, category: str=None, lang: str=None, safesearch: str=None, offset: int=None, count: int=None, countrycode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find trending news topics."
    format: Raw or HTML.
        sortby: The order to return in. Ex. Date, Relevance
        category: The category of articles to return. Ex. Sports, Entertainment, Business, Politics, World, ScienceAndTechnology, LifeStyle, China, Education, Military, RealEstate, Society, Health, Sports_NBA, Products, US_Northeast, etc.
        lang: The preferred result language. ISO 639-1 2-letter language code. 
        safesearch: Filter adult content. Ex. Off, Moderate, Strict.
        offset: The number of news articles to skip before returning.
        count: The number of articles to return. 100 Max. Increment offset by count to provide pagination to users. 
        countrycode: A two character country code where the results originate from. 
        
    """
    url = f"https://news-search7.p.rapidapi.com/trending"
    querystring = {}
    if format:
        querystring['format'] = format
    if sortby:
        querystring['sortBy'] = sortby
    if category:
        querystring['category'] = category
    if lang:
        querystring['lang'] = lang
    if safesearch:
        querystring['safeSearch'] = safesearch
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    if countrycode:
        querystring['countryCode'] = countrycode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-search7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, format: str=None, sortby: str=None, safesearch: str=None, lang: str=None, offset: int=None, count: int=None, countrycode: str=None, category: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news articles related to query."
    q: Search query string. 
        format: Raw or HTML.
        sortby: The order to return in. Ex. Date, Relevance
        safesearch: Filter adult content. Ex. Off, Moderate, Strict.
        lang: The preferred result language. ISO 639-1 2-letter language code. 
        offset: The number of news articles to skip before returning.
        count: The number of articles to return. 100 Max. Increment offset by count to provide pagination to users. 
        countrycode: A two character country code where the results originate from. 
        category: The category of articles to return. Ex. Sports, Entertainment, Business, Politics, World, ScienceAndTechnology, LifeStyle, China, Education, Military, RealEstate, Society, Health, Sports_NBA, Products, US_Northeast, etc.
        
    """
    url = f"https://news-search7.p.rapidapi.com/search"
    querystring = {'q': q, }
    if format:
        querystring['format'] = format
    if sortby:
        querystring['sortBy'] = sortby
    if safesearch:
        querystring['safeSearch'] = safesearch
    if lang:
        querystring['lang'] = lang
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    if countrycode:
        querystring['countryCode'] = countrycode
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-search7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def category(format: str=None, sortby: str=None, safesearch: str=None, countrycode: str=None, category: str=None, offset: int=None, count: int=None, headlinenum: int=None, lang: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Finds news relating to specific category."
    format: Raw or HTML.
        sortby: The order to return in. Ex. Date, Relevance
        safesearch: Filter adult content. Ex. Off, Moderate, Strict.
        countrycode: A two character country code where the results originate from. 
        category: The category of articles to return. Ex. Sports, Entertainment, Business, Politics, World, ScienceAndTechnology, LifeStyle, China, Education, Military, RealEstate, Society, Health, Sports_NBA, Products, US_Northeast, etc.
        offset: The number of news articles to skip before returning.
        count: The number of articles to return. 100 Max. Increment offset by count to provide pagination to users. 
        headlinenum: Number of headline articles to return.
        lang: The preferred result language. ISO 639-1 2-letter language code. 
        
    """
    url = f"https://news-search7.p.rapidapi.com/category"
    querystring = {}
    if format:
        querystring['format'] = format
    if sortby:
        querystring['sortBy'] = sortby
    if safesearch:
        querystring['safeSearch'] = safesearch
    if countrycode:
        querystring['countryCode'] = countrycode
    if category:
        querystring['category'] = category
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    if headlinenum:
        querystring['headlineNum'] = headlinenum
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-search7.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

