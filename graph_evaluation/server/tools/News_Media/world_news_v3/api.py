import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def findnews(sources: str=None, datefrom: str=None, language: str=None, dateto: str=None, searchin: str=None, category: str=None, pagesize: int=None, q: str=None, domains: str=None, locality: str=None, state: str=None, sortby: str=None, country: str=None, excludedomains: str=None, pagenumber: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns news articles according to query parameters"
    sources: A comma-separated string of identifiers (maximum 20) for the news sources or blogs you want headlines from. Use the /sources endpoint to locate these programmatically or look at the sources index.
        datefrom: A date and optional time for the oldest article allowed. This should be in ISO 8601 format (e.g. 2022-04-21 or 2022-04-21T01:42:02)
        language: The 2-letter ISO-639-1 code of the language you want to get headlines for. Possible options: ar de en es fr he it nl no pt ru se ud zh.
        dateto: A date and optional time for the newest article allowed. This should be in ISO 8601 format (e.g. 2022-04-21 or 2022-04-21T01:42:02)
        searchin: The fields to restrict your q search to: Options are title, description, content
        category: Find articles within this category
        pagesize: The number of results to return per page
        q: General search parameter
        domains: A comma-separated string of domains (eg bbc.co.uk, techcrunch.com, engadget.com) to restrict the search to.
        locality: Area within state
        state: Region within country
        sortby: The order to sort the articles in. Possible options: relevancy, popularity, publishedAt. relevancy = articles more closely related to q come first. popularity = articles from popular sources and publishers come first. publishedAt = newest articles come first. Default: publishedAt
        country: Origin of feed articles
        excludedomains: A comma-separated string of domains (eg bbc.co.uk, techcrunch.com, engadget.com) to remove from the results.
        pagenumber: Use this to page through results
        
    """
    url = f"https://world-news2.p.rapidapi.com/all-news"
    querystring = {}
    if sources:
        querystring['sources'] = sources
    if datefrom:
        querystring['dateFrom'] = datefrom
    if language:
        querystring['language'] = language
    if dateto:
        querystring['dateTo'] = dateto
    if searchin:
        querystring['searchIn'] = searchin
    if category:
        querystring['category'] = category
    if pagesize:
        querystring['pageSize'] = pagesize
    if q:
        querystring['q'] = q
    if domains:
        querystring['domains'] = domains
    if locality:
        querystring['locality'] = locality
    if state:
        querystring['state'] = state
    if sortby:
        querystring['sortBy'] = sortby
    if country:
        querystring['country'] = country
    if excludedomains:
        querystring['excludeDomains'] = excludedomains
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-news2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcategories(pagenumber: int=None, pagesize: int=None, q: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the news categories we support"
    pagenumber: Use this to page through results
        pagesize: The number of results to return per page
        q: Search by slug of category name
        
    """
    url = f"https://world-news2.p.rapidapi.com/categories"
    querystring = {}
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    if pagesize:
        querystring['pageSize'] = pagesize
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-news2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

