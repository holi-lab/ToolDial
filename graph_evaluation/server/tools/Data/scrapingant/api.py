import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_html_scraping(url: str, js_snippet: str=None, proxy_country: str=None, response_format: str=None, cookies: str=None, return_text: bool=None, wait_for_selector: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch raw HTML from any website using GET request"
    url: The encoded URL you want to fetch.
IMPORTANT NOTE: there is **NO** need to encode it while using from RapidAPI request builder
        js_snippet: A Base64 encoded JavaScript snippet that is run once the page is loaded.
        proxy_country: Proxy country.
Please, select one of the following:
-        'AE'    United Arab Emirates (the)
-        'BR'    Brasilia
-        'CN'    China
-        'DE'    Germany
-        'ES'    Spain
-        'FR'     France
-        'GB'   United Kingdom (the)
-        'HK'   Hong Kong
-        'IN'    India
-        'IT'    Italy
-        'IL'     Israel
-        'JP'   Japan
-        'NL'   Netherlands (the)
-        'RU'   Russia
-        'SA'   Saudi Arabia
-        'US'   USA 
        response_format: Defines the response format.
*html* by default.
When *json* used allows receiving a rich response with cookies and other useful information
        cookies: Send custom cookies to the page you want to scrape. We currently only handle name and value of custom cookies. If you want to set multiple cookies just separate cookies with ';'. Example: 'cookiename1=cookievalue1;cookiename2=cookievalue_2'
        return_text: Fetch this page and return text only or full HTML
        wait_for_selector: Valid CSS selector to wait while page load. ScrapingAnt will wait for this selector DOM element appearance and then return the result.
        
    """
    url = f"https://scrapingant.p.rapidapi.com/get"
    querystring = {'url': url, }
    if js_snippet:
        querystring['js_snippet'] = js_snippet
    if proxy_country:
        querystring['proxy_country'] = proxy_country
    if response_format:
        querystring['response_format'] = response_format
    if cookies:
        querystring['cookies'] = cookies
    if return_text:
        querystring['return_text'] = return_text
    if wait_for_selector:
        querystring['wait_for_selector'] = wait_for_selector
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrapingant.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

