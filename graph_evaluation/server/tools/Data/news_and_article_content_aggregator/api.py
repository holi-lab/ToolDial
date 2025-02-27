import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def parse_content_from_source(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get parsed content from any link. get structured results for any webpage."
    
    """
    url = f"https://news-and-article-content-aggregator1.p.rapidapi.com/parser"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-and-article-content-aggregator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_news_article_content(categorycode: str, lastkey: str, size: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get News & Article content by category. pass any of the supported category code to get the article and news content from wide range of sources.
		
		Supported content categories
		
		**Stocks, Comics, World News, Sports, Students Science, Cryptocurrency, Children Book, Startup, Technology, Finance News, Cartoon, Mutual Funds, English Learning, Kids Learning, and more**"
    
    """
    url = f"https://news-and-article-content-aggregator1.p.rapidapi.com/content"
    querystring = {'categoryCode': categorycode, 'lastKey': lastkey, 'size': size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-and-article-content-aggregator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_categories(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all the supported categories for the content."
    
    """
    url = f"https://news-and-article-content-aggregator1.p.rapidapi.com/categories"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-and-article-content-aggregator1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

