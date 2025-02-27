import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def repost(article_url: str, title: str=None, author_url: str=None, author_name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    article_url: URL of the article or blog post.
        title: Title of the telegraph post.
        author_url: URL of the author.
        author_name: Author of the telegraph post.
        
    """
    url = f"https://telegrapher.p.rapidapi.com/repost"
    querystring = {'article_url': article_url, }
    if title:
        querystring['title'] = title
    if author_url:
        querystring['author_url'] = author_url
    if author_name:
        querystring['author_name'] = author_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "telegrapher.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

