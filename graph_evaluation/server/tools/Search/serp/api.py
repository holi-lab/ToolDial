import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(q: str, hl: str='en', num: int=10, proxy_location: str='California, United States', domain: str='google.com', device: str='desktop', include_html: bool=None, gl: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allows you to search Google based on some certain parameters"
    q: This is the search query as you'd have in a regular Google search. Learn more:  [https://moz.com/blog/the-ultimate-guide-to-the-google-search-parameters](https://moz.com/blog/the-ultimate-guide-to-the-google-search-parameters).
        hl: Web interface language. Autodetected from the `domain` parameter if not supplied. Supported values are: [https://serptools.cyclic.app/google_languages](https://serptools.cyclic.app/google_languages)
        num: Number of search results. Values can be 1 - 100
        proxy_location: These are the supported locations. We offer city-level geotargeting, making it possible to narrow down your queries. See all the supported locations here: [https://serptools.cyclic.app/proxy_locations](https://serptools.cyclic.app/proxy_locations)
        domain: Specify the type of Google domain to be used. This defaults to `google.com`. See all supported domains here: [https://serptools.cyclic.app/domains](https://serptools.cyclic.app/domains)
        device: Specify the kind of device to be used for the search. This defaults to `desktop`
        include_html: Include raw html in response. This is good for debugging. It defaults to false
        gl: Country code. Autodetected from the `domain` parameter if not supplied. Supported values are: [https://serptools.cyclic.app/google_countries](https://serptools.cyclic.app/google_countries)
        
    """
    url = f"https://serp5.p.rapidapi.com/search"
    querystring = {'q': q, }
    if hl:
        querystring['hl'] = hl
    if num:
        querystring['num'] = num
    if proxy_location:
        querystring['proxy_location'] = proxy_location
    if domain:
        querystring['domain'] = domain
    if device:
        querystring['device'] = device
    if include_html:
        querystring['include_html'] = include_html
    if gl:
        querystring['gl'] = gl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serp5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check service availability. We offer 99.999% SLA."
    
    """
    url = f"https://serp5.p.rapidapi.com/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "serp5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

