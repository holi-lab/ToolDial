import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def a_chapter_by_its_slug(slug: str, provider: str, webtoon: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make request to fetch a specific chapter information for a specific webtoon from a specific provider."
    slug: Specify the chapter's slug. See /chapters for the chapter list.
        provider: Specify the webtoon provider's slug. See /providers for the provider list.
        webtoon: Specify the webtoon's slug. See /webtoons for the webtoon list.
        
    """
    url = f"https://manga-scrapper.p.rapidapi.com/chapters/{slug}"
    querystring = {'provider': provider, 'webtoon': webtoon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "manga-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_result_for_webtoon_s_query(q: str, provider: str=None, size: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make request to find webtoons based on the provided query."
    q: Specify the search query
        provider: Specify the webtoon provider's slug to get a more refined results. See /providers for the provider list.
        size: Number of search result to show between 1 - 20
        
    """
    url = f"https://manga-scrapper.p.rapidapi.com/search"
    querystring = {'q': q, }
    if provider:
        querystring['provider'] = provider
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "manga-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pass_a_cloudflare_protected_site(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make a request to fetch a website protected with Cloudflare's IUAM (I'm Under Attack Mode). This endpoint returns HTML document ready for data scraping purposes."
    
    """
    url = f"https://manga-scrapper.p.rapidapi.com/crawler"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "manga-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_chapters_updates(day: int, provider: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make request to fetch latest chapters updates up to 7 days ago."
    
    """
    url = f"https://manga-scrapper.p.rapidapi.com/updates"
    querystring = {'day': day, }
    if provider:
        querystring['provider'] = provider
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "manga-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chapters_list_all(webtoon: str, provider: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make request to fetch chapter collection for a specific webtoon from a specific provider."
    
    """
    url = f"https://manga-scrapper.p.rapidapi.com/chapters/all"
    querystring = {'webtoon': webtoon, 'provider': provider, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "manga-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chapters_list_paginated(provider: str, webtoon: str, limit: int, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make request to fetch chapter collection for a specific webtoon from a specific provider."
    provider: Specify the webtoon provider' slug. See /providers for the provider list.
        webtoon: Specify the webtoon's slug. See /webtoons for the webtoon list.
        limit: Number of results per page, between 1 - 20.
        page: Specify the page to fetch.
        
    """
    url = f"https://manga-scrapper.p.rapidapi.com/chapters"
    querystring = {'provider': provider, 'webtoon': webtoon, 'limit': limit, 'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "manga-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def providers_list_all(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make request to fetch webtoon providers list."
    
    """
    url = f"https://manga-scrapper.p.rapidapi.com/providers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "manga-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def a_webtoon_by_its_slug(provider: str, slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make request to fetch a specific webtoon information from a specific provider."
    provider: Specify the webtoon provider' slug. See /providers for the provider list.
        slug: Specify the webtoon's slug. See /webtoons for the webtoon list.
        
    """
    url = f"https://manga-scrapper.p.rapidapi.com/webtoons/{slug}"
    querystring = {'provider': provider, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "manga-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webtoons_list_all(provider: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make request to fetch webtoon's series collection from a specific provider."
    
    """
    url = f"https://manga-scrapper.p.rapidapi.com/webtoons/all"
    querystring = {'provider': provider, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "manga-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def webtoons_list_paginated(provider: str, page: int, limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make request to fetch webtoon's series collection from a specific provider."
    provider: Specify the webtoon provider' slug. See /providers for the provider list.
        page: Specify the page to fetch.
        limit: Number of results per page, between 1 - 20.
        
    """
    url = f"https://manga-scrapper.p.rapidapi.com/webtoons"
    querystring = {'provider': provider, 'page': page, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "manga-scrapper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

