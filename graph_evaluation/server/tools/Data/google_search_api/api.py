import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_result(gl: str, page: int, query: str, hl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Google Search API provides a powerful way to retrieve a wide range of information, including search engine rankings, news articles, and tweets. This versatile tool allows developers to access and integrate various types of data from Google's search index, enhancing applications with up-to-date and relevant content. By utilizing the Google Search API, you can effortlessly gather and display search results, monitor news trends, and showcase real-time Twitter posts on your platform. This forum post offers an explanation of how the Google Search API can be harnessed to extract and showcase search rankings, news articles, tweets, and more."
    
    """
    url = f"https://google-search-api8.p.rapidapi.com/"
    querystring = {'gl': gl, 'page': page, 'query': query, 'hl': hl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-search-api8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

