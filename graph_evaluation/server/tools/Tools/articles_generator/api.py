import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_generated_article_by_id(article_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When using Generate New Article Endpoint, the article is saved, and can be accessed by the provided id"
    article_id: Article ID that was outputted by the generate new article endpoint
        
    """
    url = f"https://articles-generator.p.rapidapi.com/articles/get/{article_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "articles-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

