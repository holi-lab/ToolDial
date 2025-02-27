import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def image_search(keyword: str, page: str='0', hl: str='en', safe: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Runs a standard image search and parses the output to a clean json object."
    
    """
    url = f"https://google-search-results-scraper.p.rapidapi.com/v1/scrapeGoogleImages"
    querystring = {'keyword': keyword, }
    if page:
        querystring['page'] = page
    if hl:
        querystring['hl'] = hl
    if safe:
        querystring['safe'] = safe
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search-results-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_news(region: str='US', hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current top news by region and language as a clean json object."
    
    """
    url = f"https://google-search-results-scraper.p.rapidapi.com/v1/scrapeGoogleTopNews"
    querystring = {}
    if region:
        querystring['region'] = region
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search-results-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def web_search(keyword: str, hl: str='en', gl: str='us', page: str='0', parse_ads: bool=None, safe: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Runs a standard google web search and parses the results returning a clean json object"
    hl: Two letter language code, defaults to english (en) for other codes see https://developers.google.com/admin-sdk/directory/v1/languages
        gl: Sets the country from which the search is run, defaults to US
        page: The page of the search results, defaults to 0
        parse_ads: A boolean for returning ad results which will be labeled \"is_sponsored\":true. Defaults to true.
        
    """
    url = f"https://google-search-results-scraper.p.rapidapi.com/v1/scrapeGoogle"
    querystring = {'keyword': keyword, }
    if hl:
        querystring['hl'] = hl
    if gl:
        querystring['gl'] = gl
    if page:
        querystring['page'] = page
    if parse_ads:
        querystring['parse_ads'] = parse_ads
    if safe:
        querystring['safe'] = safe
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search-results-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

