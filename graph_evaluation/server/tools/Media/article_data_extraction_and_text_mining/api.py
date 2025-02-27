import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_1_card(url: str, js_timeout: int=30, js: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract a preview of an article (article card). This is faster than the extract endpoint as it doesn't do any in-depth analysis of the content of the article, and instead mostly relies on its meta tags."
    url: URL of article to preview.

        js_timeout: when JavaScript is enabled, indicates how many seconds the API should wait for the JS interpreter before starting the extraction.

        js: indicates whether to execute JavaScript or not.

        
    """
    url = f"https://lexper.p.rapidapi.com/v1.1/card"
    querystring = {'url': url, }
    if js_timeout:
        querystring['js_timeout'] = js_timeout
    if js:
        querystring['js'] = js
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lexper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_1_extract(url: str, media: bool=None, js: str=None, js_timeout: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Article Extraction Endpoint"
    url: URL
        media: extract media
        js: indicates whether to execute JavaScript or not. 
        js_timeout: when JavaScript is enabled, indicates how many seconds the API should wait for the JS interpreter before starting the extraction
        
    """
    url = f"https://lexper.p.rapidapi.com/v1.1/extract"
    querystring = {'url': url, }
    if media:
        querystring['media'] = media
    if js:
        querystring['js'] = js
    if js_timeout:
        querystring['js_timeout'] = js_timeout
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lexper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

