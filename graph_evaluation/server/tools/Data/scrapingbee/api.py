import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def html_scraping(url: str, cookies: str=None, js_snippet: str=None, render_js: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch raw HTML from any website"
    url: The encoded URL you want to fetch
        cookies: Send custom cookies to the page you want to scrape. We currently only handle name and value of custom cookies. If you want to set multiple cookies just separate cookies with ';'. Example: 'cookie_name_1=cookie_value1;cookie_name_2=cookie_value_2'
        js_snippet: A base 64 encoded JavaScript snippet that is run once the page is fetched.
        render_js: Fetch this page and render JavaScript or not
        
    """
    url = f"https://scrapingbee.p.rapidapi.com/"
    querystring = {'url': url, }
    if cookies:
        querystring['cookies'] = cookies
    if js_snippet:
        querystring['js_snippet'] = js_snippet
    if render_js:
        querystring['render_js'] = render_js
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "scrapingbee.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

