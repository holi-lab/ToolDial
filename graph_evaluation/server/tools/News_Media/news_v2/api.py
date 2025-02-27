import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def topic_search(search: str, languages: str, publishedafter: str=None, publishedbefore: str=None, sortby: str=None, qfield: str=None, searchtype: str=None, skip: int=None, batchsize: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use topic-search endpoint to find news which talks about specific topics. Topics can range from specific keywords, to the name of a person or a place, or a combination of both. Results can be filtered by topics that are present in the title or text of the news, or both by default."
    search: Phrases or topics to be searched, in any of supported language.
        languages: Two letter ISO code to filter content by language. Use /languages resource to get language codes.
        publishedafter: ISO 8601 formatted Date Time value to filter the results which are published after the input date.
It must be within 30 days from the current date.
example format:
With date and time: YYYY-MM-DDTHH:MM:SS
With only Date: YYYY-MM-DD
        publishedbefore: ISO 8601 formatted Date Time value to filter the results which are published before the input date.
It must be within 30 days from the current date. If using the upper limit, it will show results going as far back as 35 days.
example format:
With date and time: YYYY-MM-DDTHH:MM:SS
With only Date: YYYY-MM-DD
        sortby: Sort the final results by relevance or publication date in descending order. Defaults to relevance.
        qfield: Field where input text would be matched Must be either title, or text.
        searchtype: Boolean search parameter applied between search terms. Must be AND or OR, defaults to OR.
        skip: For next batch of results use this with value increment of previous call batchSize value.
        batchsize: Number of news elements to be returned in each call. Defaults to 10, maximum 30.
        
    """
    url = f"https://news67.p.rapidapi.com/v2/topic-search"
    querystring = {'search': search, 'languages': languages, }
    if publishedafter:
        querystring['publishedAfter'] = publishedafter
    if publishedbefore:
        querystring['publishedBefore'] = publishedbefore
    if sortby:
        querystring['sortBy'] = sortby
    if qfield:
        querystring['qField'] = qfield
    if searchtype:
        querystring['searchType'] = searchtype
    if skip:
        querystring['skip'] = skip
    if batchsize:
        querystring['batchSize'] = batchsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news67.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def country_news(aboutcountry: str=None, onlyinternational: bool=None, languages: str=None, fromcountry: str='gb', batchsize: int=None, skip: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns news which are specific to countries. You can use different  filters to get local news about a country, international news about country, or inter-country news which discusses about the two countries."
    aboutcountry: Use this parameter to filter news which talks about a specific country.
        onlyinternational: To get only international news when getting news fromCountry, use true. Defaults to false.
        languages: Two letter ISO code to filter content by languages. Use comman separated codes to get results for  multiple languages. Use /languages resource to get language codes. Defaults to all.
        fromcountry: Use this parameter to filter news from a given country.
        batchsize: Number of news elements to be returned in each call.
        skip: For next batch of results use this with value increment of previous call batchSize value.
        
    """
    url = f"https://news67.p.rapidapi.com/v2/country-news"
    querystring = {}
    if aboutcountry:
        querystring['aboutCountry'] = aboutcountry
    if onlyinternational:
        querystring['onlyInternational'] = onlyinternational
    if languages:
        querystring['languages'] = languages
    if fromcountry:
        querystring['fromCountry'] = fromcountry
    if batchsize:
        querystring['batchSize'] = batchsize
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news67.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crypto_news(publishedafter: str=None, languages: str=None, batchsize: int=None, sentiment: str=None, token: str=None, skip: int=None, sortorder: str=None, publishedbefore: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest news and topics about Blockchain & Cryptocurrencies. Use sentiments to follow only positive news. Use token filter to get news about token that you want to follow."
    publishedafter: Date field to get news published after input date. Must be within last 30 days from current day to  30 mins ago from current time in UTC. 
Formatted as YYYY-MM-DDTHH:MM:SS
Default to last 30th day.
        languages: Two letter ISO language code to filter the news. Multiple languages can be selected by passing a comma separated string.
Default is all.
        batchsize: Use this parameter to limit the number of news per result. 
Default to 15, maximum 30
        sentiment: Use sentiment value of "positive" or "negative" to filter out content.
Default to both
        token: Enter token symbol to get news specific to that currency.
Default is all tokens we process.
        skip: Use this parameter for pagination to get next batch of results.
Default to 0
        sortorder: Sort the news output by "latest" news first or "oldest" news first based on captured publication date using this parameter.
Defaults to latest.
        publishedbefore: Date field to get news published before input date. Must be within last 30 days from current day to  30 mins ago from current time in UTC. 
Formatted as YYYY-MM-DDTHH:MM:SS
Default to last 30th day.
        
    """
    url = f"https://news67.p.rapidapi.com/v2/crypto"
    querystring = {}
    if publishedafter:
        querystring['publishedAfter'] = publishedafter
    if languages:
        querystring['languages'] = languages
    if batchsize:
        querystring['batchSize'] = batchsize
    if sentiment:
        querystring['sentiment'] = sentiment
    if token:
        querystring['token'] = token
    if skip:
        querystring['skip'] = skip
    if sortorder:
        querystring['sortOrder'] = sortorder
    if publishedbefore:
        querystring['publishedBefore'] = publishedbefore
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news67.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def catergories_info(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of IPTC media category labels and code o filter our news feed."
    
    """
    url = f"https://news67.p.rapidapi.com/v2/categories-info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news67.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_feed(skip: int=None, sentiment: str=None, sources: str=None, sortorder: str=None, languages: str=None, batchsize: int=None, categorycode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint fetches latest news published. This endpoint can be used to get a continuous  feed of news on your dashboard. Various filter can be applied to get the feed according to requirements. The topics are made available as soon as our pipeline finihes processing them, thus provding near real time output."
    skip: For next batch of results use this with value increment of previous call batchSize value. Default is zero.
        sentiment: Filter the feed for news with either positive or negative sentiments. Defaults to both
        sources: FIlter feed of news from different source/sources. Takes comma separated values.
        sortorder: Sort the final result in descending or ascending order of publication by passing  'oldest' or 'latest' respectively, defaults to oldest. 
        languages: Two letter ISO code to filter content by language. Use /languages resource to get language codes. Defaults to all.
        batchsize: Number of news elements to be returned in each call. Defaults to 10, maximum 30
        categorycode: Filter feed of news based on IPTC categories code. Use /categories-info resource to get the values.
        
    """
    url = f"https://news67.p.rapidapi.com/v2/feed"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if sentiment:
        querystring['sentiment'] = sentiment
    if sources:
        querystring['sources'] = sources
    if sortorder:
        querystring['sortOrder'] = sortorder
    if languages:
        querystring['languages'] = languages
    if batchsize:
        querystring['batchSize'] = batchsize
    if categorycode:
        querystring['categoryCode'] = categorycode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news67.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def world_trending(skip: str=None, top: str=None, languages: str=None, relatednewslimit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trending news topics from across the world with this endpoint. We determine trending news / / topics by considering unique sources, and nationalities which are talking about the same topic, / / along with various languages the topic is published in. Thus providing an unbiased list topics which are / / being talked about across the world."
    skip: For next batch of results use this with value increment of previous call top value.
        top: Limit number of trending topics that are returned. Defaults to 10.
        languages: Two letter ISO code to filter content by languages. Use comman separated codes to get results for  multiple languages. Use /languages resource to get language codes. Defaults to all.
        relatednewslimit: Limit the number of news that are returned inside a cluster.
        
    """
    url = f"https://news67.p.rapidapi.com/v2/trending"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if top:
        querystring['top'] = top
    if languages:
        querystring['languages'] = languages
    if relatednewslimit:
        querystring['relatedNewsLimit'] = relatednewslimit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news67.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countries_info(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of two letter country codes"
    
    """
    url = f"https://news67.p.rapidapi.com/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news67.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def language_info(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get two letter language codes for languages supported by us."
    
    """
    url = f"https://news67.p.rapidapi.com/languages"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news67.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

