import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_article_parse(url: str, word_per_minute: int=300, desc_truncate_len: int=210, desc_len_min: int=180, content_len_min: int=200, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract main article and meta data from a news entry or blog post."
    
    """
    url = f"https://article-extractor2.p.rapidapi.com/article/parse"
    querystring = {'url': url, }
    if word_per_minute:
        querystring['word_per_minute'] = word_per_minute
    if desc_truncate_len:
        querystring['desc_truncate_len'] = desc_truncate_len
    if desc_len_min:
        querystring['desc_len_min'] = desc_len_min
    if content_len_min:
        querystring['content_len_min'] = content_len_min
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "article-extractor2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

