import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def videogames_news_search(query: str, page: int=None, to_date: str=None, sort_by: str=None, from_date: str=None, per_page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns search results for current and historic news from top sources"
    query: The query text to search
        page: Use this to page through the results
        to_date: A date for the newest article (ISO 8601 format, like: 2022-03-15)
        sort_by: Sort order that will be used for ordering the results
        from_date: A date for the oldest article (ISO 8601 format, like: 2022-03-15)
        per_page: The number of results to return per page
        
    """
    url = f"https://videogames-news2.p.rapidapi.com/videogames_news/search_news"
    querystring = {'query': query, }
    if page:
        querystring['page'] = page
    if to_date:
        querystring['to_date'] = to_date
    if sort_by:
        querystring['sort_by'] = sort_by
    if from_date:
        querystring['from_date'] = from_date
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "videogames-news2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def videogames_news_recent_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the most recent video games news from top sources
		10 news items will be returned per request"
    
    """
    url = f"https://videogames-news2.p.rapidapi.com/videogames_news/recent"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "videogames-news2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

