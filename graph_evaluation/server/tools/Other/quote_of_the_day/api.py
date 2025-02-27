import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def url_api_wpquoteoftheday_com_quote_topic_id(topic_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns random quote with necessary topic"
    
    """
    url = f"https://syqel-quote-of-the-day-v1.p.rapidapi.com/url: api.wpquoteoftheday.com/quote/{topic_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "syqel-quote-of-the-day-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def url_api_wpquoteoftheday_com_category(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all available topics with id and name"
    
    """
    url = f"https://syqel-quote-of-the-day-v1.p.rapidapi.com/url: api.wpquoteoftheday.com/category"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "syqel-quote-of-the-day-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

