import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_api_scrape_article(url: str, format: str='html', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    format: Returned format of article content. html returns the article content as HTML, md returns the article content as markdown, and text returns the article content as plain text. Defaults to html.
        
    """
    url = f"https://news-article-extractor1.p.rapidapi.com/api/scrape_article"
    querystring = {'url': url, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "news-article-extractor1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

