import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def summarize(url: str, lang: str=None, html: bool=None, length: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Summarizes the article after extracting it from the specified url."
    lang: Target language to translate summary into
        html: pass TRUE if you want to convert new line symbols in API response text with html paragraph tags 
        length: Length in paragraphs. This parameter might be ignored for a very long articles.
        
    """
    url = f"https://article-extractor-and-summarizer.p.rapidapi.com/summarize"
    querystring = {'url': url, }
    if lang:
        querystring['lang'] = lang
    if html:
        querystring['html'] = html
    if length:
        querystring['length'] = length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "article-extractor-and-summarizer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extract(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract an article body and markdown version of the body."
    
    """
    url = f"https://article-extractor-and-summarizer.p.rapidapi.com/extract"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "article-extractor-and-summarizer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

