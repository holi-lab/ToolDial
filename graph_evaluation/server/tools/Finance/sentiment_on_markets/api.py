import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def article_sentiment(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return sentiment of article by given id."
    
    """
    url = f"https://sentiment-on-markets1.p.rapidapi.com/rapid-api/articles/{is_id}/sentiment"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sentiment-on-markets1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def article_summary(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return summary of article by given id."
    
    """
    url = f"https://sentiment-on-markets1.p.rapidapi.com/rapid-api/articles/{is_id}/summary"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sentiment-on-markets1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles_list(page: str='1', limit: int=10, all_tickers_mentions: bool=None, date_to: str='2022-08-31', tickers: str='AAPL', domains: str='finance.yahoo.com, seekingalpha.com', sort: str=None, date_from: str='2022-07-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return list of articles."
    all_tickers_mentions: Only in conjunction with 'tickers' filter. By default, only articles specifically written about selected tickers are included. With this option enabled, all related articles are included, even when ticker is only mentioned in content.
        date_to: Only return articles older than this date. Format %yyyy-%mm-dd.
        tickers: Only return articles related to specified tickers. Specify list of tickers ID, separated by ',' (comma).
        sort: Sort feed chronologically from most recent to oldest or vise versa.
Sorting is based on value of attribute 'published_at'.
Default is DESC.
        date_from: Only return articles newer than this date. Format %yyyy-%mm-dd.
Defult value is last 14 days
        
    """
    url = f"https://sentiment-on-markets1.p.rapidapi.com/rapid-api/articles"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    if all_tickers_mentions:
        querystring['all_tickers_mentions'] = all_tickers_mentions
    if date_to:
        querystring['date_to'] = date_to
    if tickers:
        querystring['tickers'] = tickers
    if domains:
        querystring['domains'] = domains
    if sort:
        querystring['sort'] = sort
    if date_from:
        querystring['date_from'] = date_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sentiment-on-markets1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

