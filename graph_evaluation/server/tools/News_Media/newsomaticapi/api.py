import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def top_worldwide_news(country: str=None, to: str=None, is_from: str=None, excludedomains: str=None, pagesize: int=None, sortby: str=None, qintitle: str=None, page: int=None, domains: str=None, language: str=None, q: str='bitcoin', category: str=None, sources: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get most popular worldwide news info returned in JSON format."
    country: The 2-letter ISO-639-1 code of the country you want to get headlines for. Possible options: ae ar at au be bg cr ca ch co cu cz de eg en es fr gb gr hk hu id ie in is it jp kr lt lv ma mx my ng nl np nz pe ph pk pl pt ro rs ru sa se sg si sk th tr tt tw ua uk us ve za zh

Default: all countries returned.
        to: A date and optional time for the newest article allowed. This should be in ISO 8601 format (e.g. 2021-04-11 or 2021-04-11T19:40:08)

Default: the newest according to your plan.
        is_from: A date and optional time for the oldest article allowed. This should be in ISO 8601 format (e.g. 2021-04-11 or 2021-04-11T19:40:08)

Default: the oldest according to your plan.
        excludedomains: A comma separated string of domains (eg bbc.co.uk, techcrunch.com, engadget.com) to remove from the results.
        pagesize: The number of results to return per page.

Default: 100. Maximum: 100.
        sortby: The order to sort the articles in. Possible options: relevancy, popularity, publishedAt, random.
relevancy = articles more closely related to q come first.
popularity = articles from popular sources and publishers come first.
publishedAt = newest articles come first.
random = random articles.

Default: publishedAt
        qintitle: Search query, limited only to article title (similar to the q parameter).
        page: Use this to page through the results.

Default: 1.
        domains: A comma separated string of domains (eg bbc.co.uk, techcrunch.com, engadget.com) to restrict the search to.
        language: The 2-letter ISO-639-1 code of the language you want to get headlines for. Possible options: ar bg cr cz de en es fr gr hd he hu id it jp kr lt lv nl no pl pt ro rs ru se si sk sx th tr ua ud zh

Default: all languages returned.
        q: Keywords or phrases to search for in the article title and body. Advanced search is supported here:

Surround phrases with quotes (“) for exact match.
Prepend words or phrases that must appear with a + symbol. Eg: +bitcoin
Prepend words that must not appear with a – symbol. Eg: -bitcoin
The complete value for q must be URL-encoded.

Keywords or phrases to search for in the article title only.

Advanced search is supported here:

Surround phrases with quotes (“) for exact match.
Prepend words or phrases that must appear with a + symbol. Eg: +bitcoin
Prepend words that must not appear with a – symbol. Eg: -bitcoin
        category: Find sources that display news of this category. Possible options: business entertainment general health science sports technology. Default: all categories.


        sources: A comma separated string of identifiers (maximum 10) for the news sources or blogs you want headlines from. Use the /sources endpoint to locate these programmatically or look at the sources index.
        
    """
    url = f"https://newsomaticapi.p.rapidapi.com/top"
    querystring = {}
    if country:
        querystring['country'] = country
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    if excludedomains:
        querystring['excludeDomains'] = excludedomains
    if pagesize:
        querystring['pageSize'] = pagesize
    if sortby:
        querystring['sortBy'] = sortby
    if qintitle:
        querystring['qInTitle'] = qintitle
    if page:
        querystring['page'] = page
    if domains:
        querystring['domains'] = domains
    if language:
        querystring['language'] = language
    if q:
        querystring['q'] = q
    if category:
        querystring['category'] = category
    if sources:
        querystring['sources'] = sources
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newsomaticapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sources_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the subset of news publishers that top headlines are available from. It’s mainly a convenience endpoint that you can use to keep track of the publishers available on the API."
    
    """
    url = f"https://newsomaticapi.p.rapidapi.com/sources"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newsomaticapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_worldwide_news(category: str=None, to: str=None, page: int=None, language: str=None, excludedomains: str=None, qintitle: str='news', q: str='bitcoin', is_from: str=None, sources: str=None, domains: str=None, sortby: str=None, country: str=None, pagesize: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get worldwide news info returned in JSON format."
    category: Find sources that display news of this category. Possible options: business entertainment general health science sports technology. Default: all categories.


        to: A date and optional time for the newest article allowed. This should be in ISO 8601 format (e.g. 2021-04-11 or 2021-04-11T19:40:08)

Default: the newest according to your plan.
        page: Use this to page through the results.

Default: 1.
        language: The 2-letter ISO-639-1 code of the language you want to get headlines for. Possible options: ar bg cr cz de en es fr gr hd he hu id it jp kr lt lv nl no pl pt ro rs ru se si sk sx th tr ua ud zh

Default: all languages returned.
        excludedomains: A comma separated string of domains (eg bbc.co.uk, techcrunch.com, engadget.com) to remove from the results.
        qintitle: Search query, limited only to article title (similar to the q parameter).
        q: Keywords or phrases to search for in the article title and body. Advanced search is supported here:

Surround phrases with quotes (“) for exact match.
Prepend words or phrases that must appear with a + symbol. Eg: +bitcoin
Prepend words that must not appear with a – symbol. Eg: -bitcoin
The complete value for q must be URL-encoded.

Keywords or phrases to search for in the article title only.

Advanced search is supported here:

Surround phrases with quotes (“) for exact match.
Prepend words or phrases that must appear with a + symbol. Eg: +bitcoin
Prepend words that must not appear with a – symbol. Eg: -bitcoin
        is_from: A date and optional time for the oldest article allowed. This should be in ISO 8601 format (e.g. 2021-04-11 or 2021-04-11T19:40:08)

Default: the oldest according to your plan.
        sources: A comma separated string of identifiers (maximum 10) for the news sources or blogs you want headlines from. Use the /sources endpoint to locate these programmatically or look at the sources index.
        domains: A comma separated string of domains (eg bbc.co.uk, techcrunch.com, engadget.com) to restrict the search to.
        sortby: The order to sort the articles in. Possible options: relevancy, popularity, publishedAt, random.
relevancy = articles more closely related to q come first.
popularity = articles from popular sources and publishers come first.
publishedAt = newest articles come first.
random = random articles.

Default: publishedAt
        country: The 2-letter ISO-639-1 code of the country you want to get headlines for. Possible options: ae ar at au be bg cr ca ch co cu cz de eg en es fr gb gr hk hu id ie in is it jp kr lt lv ma mx my ng nl np nz pe ph pk pl pt ro rs ru sa se sg si sk th tr tt tw ua uk us ve za zh

Default: all countries returned.
        pagesize: The number of results to return per page.

Default: 100. Maximum: 100.
        
    """
    url = f"https://newsomaticapi.p.rapidapi.com/all"
    querystring = {}
    if category:
        querystring['category'] = category
    if to:
        querystring['to'] = to
    if page:
        querystring['page'] = page
    if language:
        querystring['language'] = language
    if excludedomains:
        querystring['excludeDomains'] = excludedomains
    if qintitle:
        querystring['qInTitle'] = qintitle
    if q:
        querystring['q'] = q
    if is_from:
        querystring['from'] = is_from
    if sources:
        querystring['sources'] = sources
    if domains:
        querystring['domains'] = domains
    if sortby:
        querystring['sortBy'] = sortby
    if country:
        querystring['country'] = country
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "newsomaticapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

