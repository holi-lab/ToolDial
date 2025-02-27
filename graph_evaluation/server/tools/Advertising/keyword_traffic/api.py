import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def keyword_traffic_test(match_type: str, keywords: str, traffic_targets: str, location_criterion_id: int=2840, language: str='en', language_criterion_id: int=1000, location: str='us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides Google Ads and Bing Ads traffic information."
    
    """
    url = f"https://keyword-traffic.p.rapidapi.com/"
    querystring = {'match_type': match_type, 'keywords': keywords, 'traffic_targets': traffic_targets, }
    if location_criterion_id:
        querystring['location_criterion_id'] = location_criterion_id
    if language:
        querystring['language'] = language
    if language_criterion_id:
        querystring['language_criterion_id'] = language_criterion_id
    if location:
        querystring['location'] = location
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "keyword-traffic.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

