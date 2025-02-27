import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def article_api(url: str, timeout: int=15000, paging: bool=None, fields: str='text,html,images(pixelHeight,pixelWidth)', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Article API is used to extract clean article text from news articles, blog posts, and other text-heavy web pages."
    url: URL to extract article from (URLEncoded)
        timeout: Specify a value in milliseconds (e.g., &timeout=15000) to override the default API timeout of 5000ms.
        paging: Send paging=false to disable automatic concatenation of multi-page articles.
        fields: Send in a list of comma-separated fieldnames to override default field output in the response.
        
    """
    url = f"https://diffbot-diffbot.p.rapidapi.com/v2/article"
    querystring = {'url': url, }
    if timeout:
        querystring['timeout'] = timeout
    if paging:
        querystring['paging'] = paging
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diffbot-diffbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def frontpage(url: str, timeout: str='15000', format: str='json', all: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Frontpage API takes in a multifaceted “homepage” and returns individual page elements."
    url: Frontpage URL from which to extract items
        timeout: Specify a value in milliseconds (e.g., &timeout=15000) to override the default API timeout of 5000ms.
        format: Format the response output in xml (default) or json
        all: Returns all content from page, including navigation and similar links that the Diffbot visual processing engine considers less important / non-core.
        
    """
    url = f"https://diffbot-diffbot.p.rapidapi.com/frontpage"
    querystring = {'url': url, }
    if timeout:
        querystring['timeout'] = timeout
    if format:
        querystring['format'] = format
    if all:
        querystring['all'] = all
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diffbot-diffbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_api(url: str, timeout: int=15000, fields: str='products(offerPrice,regularPrice)', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Product API retrieves full product data from any e-commerce shopping page."
    url: URL of the page to process.
        timeout: Specify a value in milliseconds (e.g., &timeout=15000) to override the default API timeout of 5000ms.
        fields: Send in a list of comma-separated fieldnames to override default field output in the response.
        
    """
    url = f"https://diffbot-diffbot.p.rapidapi.com/v2/product"
    querystring = {'url': url, }
    if timeout:
        querystring['timeout'] = timeout
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "diffbot-diffbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

