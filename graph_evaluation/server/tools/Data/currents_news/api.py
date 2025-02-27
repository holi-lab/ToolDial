import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latest_news(language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest news"
    language: Valid value : Supported code can be found from /v1/available/languages. Default : en English
        
    """
    url = f"https://currents-news.p.rapidapi.com/latest-news"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currents-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(start_date: str=None, end_date: str=None, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint allow you to search through ten millions of article over 14,000 large and small news sources and blogs. This includes breaking news, blog articles, forum content. This endpoint is well suited for article discovery and analysis, but can be used to retrieve articles for display, too."
    start_date: Default : current time value in UCT+0 Valid format : Date format should be YYYY-MM-ddTHH:mm:ss.ss±hh:mm, which follows the offcial standard of RFC 3339 Date and Time on the Internet
        end_date: Default : current time value in UCT+0 Valid format : Date format should be YYYY-MM-ddTHH:mm:ss.ss±hh:mm, which follows the offcial standard of RFC 3339 Date and Time on the Internet
        language: Valid value : Supported code can be found from /v1/available/languages
        
    """
    url = f"https://currents-news.p.rapidapi.com/search"
    querystring = {}
    if start_date:
        querystring['start_date'] = start_date
    if end_date:
        querystring['end_date'] = end_date
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "currents-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

