import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hot_latest_news(is_id: str, sortby: str='hot', pagenumber: str='1', numberofrow: str='5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides hot - latest news for a country (us, uk, australia, ...) or a topic (tech, travel, android, marketing, ...).
		If you want custom topic or country, please contact us."
    id: For country: country + the 2-letter ISO 3166-1 code of the country
Possible options: au, ca, ie, nz, ng, ph, sg, za, gb, us
Example: country-us, country-gb

For topic:  topic + topic name
Possible options: tech, marketing, science, football, crypto
Example: topic-tech, topic-marketing
        sortby: Possible options: hot, newest
Default: newest
        pagenumber: Use this to page through the results if the total results found is greater than the page size. Start with 1
        numberofrow: The number of results to return per page (request), 5 is the default
Basic plan maximum: 5
Pro plan maximum: 20
Ultra plan maximum: 100
        
    """
    url = f"https://hot-breaking-news-latest-news.p.rapidapi.com/exec"
    querystring = {'id': is_id, }
    if sortby:
        querystring['sortBy'] = sortby
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    if numberofrow:
        querystring['numberOfRow'] = numberofrow
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hot-breaking-news-latest-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

