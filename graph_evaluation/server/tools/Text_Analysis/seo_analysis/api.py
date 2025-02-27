import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def content_analysis(url: str, keyword: str, relatedkeywords: str='SEO Tools|Free SEO tools|online tools', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Content Analysis endpoint"
    
    """
    url = f"https://seo-analysis.p.rapidapi.com/seo-content-analysis/"
    querystring = {'url': url, 'keyword': keyword, }
    if relatedkeywords:
        querystring['relatedkeywords'] = relatedkeywords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seo-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

