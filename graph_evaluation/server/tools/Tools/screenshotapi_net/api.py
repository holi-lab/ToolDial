import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def screenshot(output: str, token: str, url: str, width: int=None, css_url: str=None, fresh: bool=None, css: str=None, full_page: bool=None, cookies: str=None, selector: str=None, accept_languages: str=None, delay: int=None, thumbnail_width: int=None, height: int=None, user_agent: str=None, ttl: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate beautiful website screenshots using our fast website screenshot API."
    
    """
    url = f"https://screenshotapi-net1.p.rapidapi.com/"
    querystring = {'output': output, 'token': token, 'url': url, }
    if width:
        querystring['width'] = width
    if css_url:
        querystring['css_url'] = css_url
    if fresh:
        querystring['fresh'] = fresh
    if css:
        querystring['css'] = css
    if full_page:
        querystring['full_page'] = full_page
    if cookies:
        querystring['cookies'] = cookies
    if selector:
        querystring['selector'] = selector
    if accept_languages:
        querystring['accept_languages'] = accept_languages
    if delay:
        querystring['delay'] = delay
    if thumbnail_width:
        querystring['thumbnail_width'] = thumbnail_width
    if height:
        querystring['height'] = height
    if user_agent:
        querystring['user_agent'] = user_agent
    if ttl:
        querystring['ttl'] = ttl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "screenshotapi-net1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

