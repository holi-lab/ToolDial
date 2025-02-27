import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def extract_news(url: str, analyze: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract a news article from a news site."
    url: The url of the news.
        analyze: Whether to analyze the news (extract entities etc.)
        
    """
    url = f"https://world-news3.p.rapidapi.com/extract-news"
    querystring = {'url': url, 'analyze': analyze, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-news3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geo_coordinates(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the geo coordinates for a location. The location can be an exact address but also just the name of a city or country."
    location: The address or name of the location, e.g. Tokyo, Japan.
        
    """
    url = f"https://world-news3.p.rapidapi.com/geo-coordinates"
    querystring = {'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-news3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_news(text: str='hurricane', news_sources: str='https://www.bbc.co.uk', location_filter: str='51.050407,13.737262,100', source_countries: str='gb,us', max_sentiment: int=None, offset: int=10, sort: str='publish-time', authors: str='John Doe', sort_direction: str='desc', latest_publish_date: str='2022-05-23 24:16:27', earliest_publish_date: str='2022-04-22 16:12:35', language: str='en', min_sentiment: int=None, entities: str='ORG:Tesla', number: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search, filter, and sort news articles."
    text: The text to match in the news content.
        news_sources: A comma-separated list of news sources from which the news should originate, e.g. https://www.bbc.co.uk
        location_filter: Filter news by radius around a certain location. Format is \"latitude,longitude,radius in kilometers\", e.g. 51.050407, 13.737262, 100
        source_countries: A comma-separated list of ISO 3166 country codes from which the news should originate, e.g. gb,us.
        max_sentiment: The maximal sentiment of the news in range [-1,1].
        offset: The number of news to skip in range [0,1000]
        sort: The sorting criteria.
        authors: A comma-separated list of author names. Only news from any of the given authors will be returned.
        sort_direction: Whether to sort ascending or descending.
        latest_publish_date: The news must have been published before this date.
        earliest_publish_date: The news must have been published after this date.
        language: The ISO 6391 language code of the news, e.g. \"en\" for English.
        min_sentiment: The minimal sentiment of the news in range [-1,1].
        entities: Filter news by entities, e.g. ORG:Tesla.
        number: The number of news to return in range [1,100]
        
    """
    url = f"https://world-news3.p.rapidapi.com/search-news"
    querystring = {}
    if text:
        querystring['text'] = text
    if news_sources:
        querystring['news-sources'] = news_sources
    if location_filter:
        querystring['location-filter'] = location_filter
    if source_countries:
        querystring['source-countries'] = source_countries
    if max_sentiment:
        querystring['max-sentiment'] = max_sentiment
    if offset:
        querystring['offset'] = offset
    if sort:
        querystring['sort'] = sort
    if authors:
        querystring['authors'] = authors
    if sort_direction:
        querystring['sort-direction'] = sort_direction
    if latest_publish_date:
        querystring['latest-publish-date'] = latest_publish_date
    if earliest_publish_date:
        querystring['earliest-publish-date'] = earliest_publish_date
    if language:
        querystring['language'] = language
    if min_sentiment:
        querystring['min-sentiment'] = min_sentiment
    if entities:
        querystring['entities'] = entities
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "world-news3.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

