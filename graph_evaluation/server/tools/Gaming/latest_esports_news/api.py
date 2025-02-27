import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dotesports_articles_with_pagination(categories: str='cs', page: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Dotesports' articles are published using pagination, so I have enabled you to target each page of articles using a `page` query string parameter. Integers greater than 0 are valid values for `page`. Higher values will return older articles.
		
		This can also be used in tandem with the `categories` query string parameter if you wish to filter out unrelated articles."
    page: Corresponds to the page of results on Dotesports' website.

For example, `?page=2` would correspond to the articles located at `https://www.dotesports.com/page/2`.
        
    """
    url = f"https://latest-esports-news.p.rapidapi.com/api/dotesports/"
    querystring = {}
    if categories:
        querystring['categories'] = categories
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latest-esports-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles_from_one_source(source: str, categories: str='valorant,league of legends', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to return the latest articles from a specified news source. It corresponds to the `source` field found in each resource.
		
		For example, to get articles from _Dexerto_, you would make a GET request to https://latest-esports-news.p.rapidapi.com/api/dexerto."
    categories: The `categories` query string parameter allows you to filter the returned list of articles by desired topics.

   A valid category is any category that exists on the news source's website. For example, Dexerto has a category called _Business_, which you could filter by.

   Most categories, however, are popular eSports titles, and these categories are shared between both news sources. Therefore, if you filter by _Valorant_, you'll find articles from both Dexerto and Dotesports in the returned list.

   You can include multiple categories in a query by separating them with a comma. In the example URL above, we would get all of the articles of `category1` or `category2`.
        
    """
    url = f"https://latest-esports-news.p.rapidapi.com/api/{source}"
    querystring = {}
    if categories:
        querystring['categories'] = categories
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latest-esports-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles_from_all_sources(categories: str='cod,cs', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is the root endpoint for the API. It will fetch articles from all the supported news sources. They will be sorted from most recently published to oldest."
    categories: The `categories` query string parameter allows you to filter the returned list of articles by desired topics.

   A valid category is any category that exists on the news source's website. For example, Dexerto has a category called _Business_, which you could filter by.

   Most categories, however, are popular eSports titles, and these categories are shared between both news sources. Therefore, if you filter by _Valorant_, you'll find articles from both Dexerto and Dotesports in the returned list.

   You can include multiple categories in a query by separating them with a comma. In the example URL above, we would get all of the articles of `category1` or `category2`.
        
    """
    url = f"https://latest-esports-news.p.rapidapi.com/api/"
    querystring = {}
    if categories:
        querystring['categories'] = categories
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "latest-esports-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

