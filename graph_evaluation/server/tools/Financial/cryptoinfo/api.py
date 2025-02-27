import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_last_5_news(content_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get last 5 actual and processed financial news"
    
    """
    url = f"https://cryptoinfo.p.rapidapi.com/api/private/latest_news/rapid_api/news/5"
    querystring = {'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptoinfo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_last_5_economic_calendar_news(content_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get last 5 actual and processed economic calendar news"
    
    """
    url = f"https://cryptoinfo.p.rapidapi.com/api/private/latest_news/rapid_api/economic-calendar-large/5"
    querystring = {'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptoinfo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_economic_calendar_news_over_a_period_of_time(content_type: str='application/json', time_start: str='2022-12-20 17:34:58+00:00', time_finish: str='2023-02-13 19:34:58+00:00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get news from one date to another. Maximum from 30 days range"
    
    """
    url = f"https://cryptoinfo.p.rapidapi.com/api/private/news_over_a_period_of_time/rapid_api/economic-calendar-large"
    querystring = {}
    if content_type:
        querystring['Content-Type'] = content_type
    if time_start:
        querystring['time_start'] = time_start
    if time_finish:
        querystring['time_finish'] = time_finish
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptoinfo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_news_over_a_period_of_time(content_type: str, time_start: str='2023-01-20 17:34:58+00:00', time_finish: str='2023-01-21 17:34:58+00:00', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get news from one date to another. Maximum from 1 day range"
    
    """
    url = f"https://cryptoinfo.p.rapidapi.com/api/private/news_over_a_period_of_time/rapid_api/news"
    querystring = {'Content-Type': content_type, }
    if time_start:
        querystring['time_start'] = time_start
    if time_finish:
        querystring['time_finish'] = time_finish
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cryptoinfo.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

