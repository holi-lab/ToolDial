import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def trending(offset: int=None, language: str='en', category: str=None, from_sources: str=None, number: int=None, has_image: bool=None, exclude_sources: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a continuously updated, rich feed of articles finely picked by a sophisticated A.I.
		
		Try it live at: https://webit.re/services/internet-search/news-search-api/tryout"
    offset: Offset to start getting results from.

For example, if you search 10 articles at a time (number=10), then you should set offset=0 for page 1, offset=10 for page 2, offset=20 for page 3 and so on.
        language: Supported languages (ISO 639-1 codes):
Supported languages (ISO 639-1 codes):
- Stable: 'en', 'bg', 'de', 'es', 'fr', 'fi', 'it', 'ja', 'nl', 'pl', 'pt', 'ro', 'ru', 'zh';
- Beta: 'ar', 'ca', 'ko', 'nb', 'sv', 'tr', 'uk';
- Alpha: 'be', 'ca', 'da', 'el', 'et', 'fa', 'ga', 'gl', 'he', 'hi', 'hr', 'hu', 'id', 'lv', 'no', 'om', 'sk', 'sr', 'tt', 'vi'.
        category: [COMING SOON (APRIL 2021) - It is currently ignored] Category to restrict articles by.
        from_sources: [PREMIUM - PRO+ plans only] Comma separated list of sources hosts to pick news from. For instance: \"nytimes.com, digitaltrends.com, ...\", etc. This will exclude any other source.
        number: Number of articles to retrieve with a single request.

Maximum allowed results number per request:
- 10 results for Free/Basic plan;
- 50 results for Pro plan;
- 50 results for Ultra plan;
- 100 results for Mega plan.

For requesting any customized quota, please contact our support team.
        has_image: Set this to \"True\" in order to get only articles having an image associated to. Default is \"False\".
        exclude_sources: [PREMIUM - PRO+ plans only] Comma separated list of sources hosts to exclude from the results. For instance: \"nytimes.com, digitaltrends.com, ...\", etc.
        
    """
    url = f"https://webit-news-search.p.rapidapi.com/trending"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if language:
        querystring['language'] = language
    if category:
        querystring['category'] = category
    if from_sources:
        querystring['from_sources'] = from_sources
    if number:
        querystring['number'] = number
    if has_image:
        querystring['has_image'] = has_image
    if exclude_sources:
        querystring['exclude_sources'] = exclude_sources
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-news-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, language: str='en', exclude_sources: str=None, category: str=None, from_sources: str=None, offset: int=0, has_image: bool=None, number: int=8, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search in a continuously updated database containing millions of articles finely crawled minute by minute from sources selected by a sophisticated A.I.
		
		Try it live at: https://webit.re/services/internet-search/news-search-api/tryout"
    q: String including the search terms to seek articles for
        language: Language to search articles for.

Supported languages (ISO 639-1 codes):
- Stable: 'en', 'bg', 'de', 'es', 'fr', 'fi', 'it', 'ja', 'nl', 'pl', 'pt', 'ro', 'ru', 'zh';
- Beta: 'ar', 'ca', 'ko', 'nb', 'sv', 'tr', 'uk';
- Alpha: 'be', 'ca', 'da', 'el', 'et', 'fa', 'ga', 'gl', 'he', 'hi', 'hr', 'hu', 'id', 'lv', 'no', 'om', 'sk', 'sr', 'tt', 'vi'.
        exclude_sources: [PREMIUM - PRO+ plans only] Comma separated list of sources hosts to exclude from the results. For instance: \"nytimes.com, digitaltrends.com, ...\", etc.
        category: [COMING SOON (APRIL 2021) - It is currently ignored] Category to restrict articles by.
        from_sources: [PREMIUM - PRO+ plans only] Comma separated list of sources hosts to pick news from. For instance: \"nytimes.com, digitaltrends.com, ...\", etc. This will exclude any other source.
        offset: Offset to start getting results from.

For example, if you search 10 articles at a time (number=10), then you should set offset=0 for page 1, offset=10 for page 2, offset=20 for page 3 and so on.
        has_image: Set this to \"True\" in order to get only articles having an image associated to. Default is \"False\".
        number: Number of articles to retrieve with a single request.

Maximum allowed results number per request:
- 10 results for Free/Basic plan;
- 50 results for Pro plan;
- 50 results for Ultra plan;
- 100 results for Mega plan.

For requesting any customized quota, please contact our support team.
        
    """
    url = f"https://webit-news-search.p.rapidapi.com/search"
    querystring = {'q': q, }
    if language:
        querystring['language'] = language
    if exclude_sources:
        querystring['exclude_sources'] = exclude_sources
    if category:
        querystring['category'] = category
    if from_sources:
        querystring['from_sources'] = from_sources
    if offset:
        querystring['offset'] = offset
    if has_image:
        querystring['has_image'] = has_image
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webit-news-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

