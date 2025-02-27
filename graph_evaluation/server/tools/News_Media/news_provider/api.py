import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_news_about_specific_topic_and_subtopic(topic: str, subtopic: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It Provides news about Specific Topic And SubTopic. 
		In order To know all available Topics and Subtopic please use the root of the endpoint."
    
    """
    url = f"https://news-provider.p.rapidapi.com/{topic}/{subtopic}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

