import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def language_list(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get valid languages for a country code, to be used with all other APIs."
    country: Country code of the country to get languages for. See [all available country codes](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
        
    """
    url = f"https://real-time-news-data.p.rapidapi.com/language-list"
    querystring = {'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-news-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topic_headlines(topic: str, lang: str='en', country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest news headlines for a topic."
    topic: Topic for which to get news headlines.

**Available topic**
- WORLD
- NATIONAL
- BUSINESS
- TECHNOLOGY
- ENTERTAINMENT
- SPORTS
- SCIENCE
- HEALTH

In addition, topic IDs are also accepted and can be taken from a News topic URL as it appears after the *topics/* path part (e.g. Elon Musk Topic - `/topics/CAAqIggKIhxDQkFTRHdvSkwyMHZNRE51ZW1ZeEVnSmxiaWdBUAE`) 
        lang: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1 alpha-2](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default**: `en`.
        country: Country code. See [all available country codes](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
**Default:** `US`.
        
    """
    url = f"https://real-time-news-data.p.rapidapi.com/topic-headlines"
    querystring = {'topic': topic, }
    if lang:
        querystring['lang'] = lang
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-news-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_headlines(country: str='US', lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest news headlines/top stories for a country."
    country: Country code. See [all available country codes](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
**Default:** `US`.
        lang: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1 alpha-2](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default**: `en`.
        
    """
    url = f"https://real-time-news-data.p.rapidapi.com/top-headlines"
    querystring = {}
    if country:
        querystring['country'] = country
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-news-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, country: str='US', lang: str='en', source: str=None, time_published: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search news articles by query with an option to limit the results to a specific time range."
    query: Search query for which to get news.
        country: Country code. See [all available country codes](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
**Default:** `US`.
        lang: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1 alpha-2](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default**: `en`.
        source: Domain of the source from which to return news articles (e.g. cnn.com).
        time_published: Find news articles published in a specific time range. 
**Default:** `anytime`
        
    """
    url = f"https://real-time-news-data.p.rapidapi.com/search"
    querystring = {'query': query, }
    if country:
        querystring['country'] = country
    if lang:
        querystring['lang'] = lang
    if source:
        querystring['source'] = source
    if time_published:
        querystring['time_published'] = time_published
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-news-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def local_headlines_geo(query: str, lang: str='en', country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get local, geo based headlines"
    query: Area, city or country to fetch news for (e.g. *London*).
        lang: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1 alpha-2](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default**: `en`.
        country: Country code. See [all available country codes](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
**Default:** `US`.
        
    """
    url = f"https://real-time-news-data.p.rapidapi.com/local-headlines"
    querystring = {'query': query, }
    if lang:
        querystring['lang'] = lang
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-news-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def topic_news_by_section(section: str, topic: str, lang: str='en', country: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get topic news article in a specific section."
    topic: Topic for which to get news headlines.

**Available topic**
- WORLD
- NATIONAL
- BUSINESS
- TECHNOLOGY
- ENTERTAINMENT
- SPORTS
- SCIENCE
- HEALTH

In addition, topic IDs are also accepted and can be taken from a News topic URL as it appears after the *topic/* path part (e.g. Elon Musk Topic - `/topics/CAAqIggKIhxDQkFTRHdvSkwyMHZNRE51ZW1ZeEVnSmxiaWdBUAE`) 
        lang: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1 alpha-2](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).
**Default**: `en`.
        country: Country code. See [all available country codes](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
**Default:** `US`.
        
    """
    url = f"https://real-time-news-data.p.rapidapi.com/topic-news-by-section"
    querystring = {'section': section, 'topic': topic, }
    if lang:
        querystring['lang'] = lang
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-news-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

