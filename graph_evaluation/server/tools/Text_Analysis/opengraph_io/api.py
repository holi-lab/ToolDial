import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def site_lookup(url: str, max_cache_age: str='432000000', full_render: bool=None, cache_ok: bool=None, accept_lang: str='en-US,en;q=0.9', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will retrieve Open Graph tags from a given URL. If it is not present, our API will infer it for you."
    url: The URL to be fetched (e.g. https://reddit.com)
        max_cache_age: This specifies the maximum age in milliseconds that a cached response should be. If not specified the value is set to 5 days. (5 days * 24 hours * 60 minutes * 60 seconds * 1000 ms = 432,000,000 ms)
        full_render: This will fully render the site using a chrome browser before parsing its contents. This is especially helpful for single page applications and JS redirects. This will slow down the time it takes to get a response by around 1.5 seconds.
        cache_ok: This will force our servers to pull a fresh version of the site being requested. By default this value is true
        accept_lang: This specifies the request language sent when requesting the url. This is useful if you want to get the site for languages other than english. The default setting for this will return an english version of a page if it exists. Note: if you specify the value auto the api will use the same language settings of your current request.
        
    """
    url = f"https://opengraph-io.p.rapidapi.com/api/1.1/sites"
    querystring = {'url': url, }
    if max_cache_age:
        querystring['max_cache_age'] = max_cache_age
    if full_render:
        querystring['full_render'] = full_render
    if cache_ok:
        querystring['cache_ok'] = cache_ok
    if accept_lang:
        querystring['accept_lang'] = accept_lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opengraph-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scrape(site: str, cache_ok: bool=None, max_cache_age: str='432000000', full_render: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Just need the raw HTML?
		
		The Scrape Site endpoint is used to scrape the HTML of a website given its URL"
    cache_ok: This will force our servers to pull a fresh version of the site being requested. By default this value is true
        max_cache_age: This specifies the maximum age in milliseconds that a cached response should be. If not specified the value is set to 5 days. (5 days * 24 hours * 60 minutes * 60 seconds * 1000 ms = 432,000,000 ms)
        full_render: This will fully render the site using a chrome browser before parsing its contents. This is especially helpful for single page applications and JS redirects. This will slow down the time it takes to get a response by around 1.5 seconds.
        
    """
    url = f"https://opengraph-io.p.rapidapi.com/api/1.1/scrape/{site}"
    querystring = {}
    if cache_ok:
        querystring['cache_ok'] = cache_ok
    if max_cache_age:
        querystring['max_cache_age'] = max_cache_age
    if full_render:
        querystring['full_render'] = full_render
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opengraph-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extract(site: str, max_cache_age: str='432000000', cache_ok: bool=None, html_elements: str='h1,h2,h3,p,title', full_render: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The extract endpoint enables you to extract information from any website by providing its URL. With this endpoint, you can extract any element you need from the website, including but not limited to the title, header elements (h1 to h5), and paragraph elements (p)."
    max_cache_age: This specifies the maximum age in milliseconds that a cached response should be. If not specified the value is set to 5 days. (5 days * 24 hours * 60 minutes * 60 seconds * 1000 ms = 432,000,000 ms)
        cache_ok: This will force our servers to pull a fresh version of the site being requested. By default this value is true
        full_render: This will fully render the site using a chrome browser before parsing its contents. This is especially helpful for single page applications and JS redirects. This will slow down the time it takes to get a response by around 1.5 seconds.
        
    """
    url = f"https://opengraph-io.p.rapidapi.com/api/1.1/extract/{site}"
    querystring = {}
    if max_cache_age:
        querystring['max_cache_age'] = max_cache_age
    if cache_ok:
        querystring['cache_ok'] = cache_ok
    if html_elements:
        querystring['html_elements'] = html_elements
    if full_render:
        querystring['full_render'] = full_render
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opengraph-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

