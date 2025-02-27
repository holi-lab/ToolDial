import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_article(u: str, n: str, a: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert an article from TDS to Markdown"
    
    """
    url = f"https://towards-data-science-article-to-markdown-blog-converter.p.rapidapi.com/get_article"
    querystring = {'u': u, 'n': n, 'a': a, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "towards-data-science-article-to-markdown-blog-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

