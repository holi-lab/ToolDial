import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def analyze_v2(url: str, category: str=None, strategy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "analyze page speed with more results information"
    category: the options: **accessibility, best-practices, performance, seo, pwa**. default: performance 
        strategy: the options: **desktop, mobile** default: empty
        
    """
    url = f"https://seo-checker2.p.rapidapi.com/analyze-v2"
    querystring = {'url': url, }
    if category:
        querystring['category'] = category
    if strategy:
        querystring['strategy'] = strategy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-checker2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def analyze(url: str, strategy: str=None, category: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "analyze page speed"
    strategy: the options: **desktop, mobile** default: empty
        category: the options: **accessibility, best-practices, performance, seo, pwa**. default: performance 
        
    """
    url = f"https://seo-checker2.p.rapidapi.com/analyze"
    querystring = {'url': url, }
    if strategy:
        querystring['strategy'] = strategy
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-checker2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def summary(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Summary SEO"
    
    """
    url = f"https://seo-checker2.p.rapidapi.com/summary"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-checker2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

