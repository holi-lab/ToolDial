import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def summarize(url: str, type: str=None, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pass in a web article URL and get a summary of the article back. Optionally you can pass in a language code you'd like the summary to be translated to."
    url: The URL of the web article to be summarized
        type: Determined if a shorter (default) or longer summary should returned. To return a slightly longer summary, pass 'expand' here.
        lang: Language the summary should be translated to. Leave blank to keep the summary in the same language as the article.
        
    """
    url = f"https://summate-it.p.rapidapi.com/summarize"
    querystring = {'url': url, }
    if type:
        querystring['type'] = type
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "summate-it.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

