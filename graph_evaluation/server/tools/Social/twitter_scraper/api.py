import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(todate: str=None, near: str=None, url: str=None, fromdate: str=None, lang: str=None, maxtweets: int=100, searchmode: str=None, searchterms: str='wikipedia', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Twitter by keyword, hashtag or URL"
    todate: Check the About tab for more info
        near: Check the About tab for more info
        url: Check the About tab for more info
        fromdate: Check the About tab for more info
        lang: Check the About tab for more info
        maxtweets: Check the About tab for more info
        searchmode: Check the About tab for more info
        searchterms: Check the About tab for more info
        
    """
    url = f"https://twitter-scraper2.p.rapidapi.com/search"
    querystring = {}
    if todate:
        querystring['toDate'] = todate
    if near:
        querystring['near'] = near
    if url:
        querystring['url'] = url
    if fromdate:
        querystring['fromDate'] = fromdate
    if lang:
        querystring['lang'] = lang
    if maxtweets:
        querystring['maxTweets'] = maxtweets
    if searchmode:
        querystring['searchMode'] = searchmode
    if searchterms:
        querystring['searchTerms'] = searchterms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twitter-scraper2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

