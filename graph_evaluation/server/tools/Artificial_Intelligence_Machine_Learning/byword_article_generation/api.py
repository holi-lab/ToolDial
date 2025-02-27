import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def write_article_from_title(title: str, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Input the title of an article that you'd like written, and this endpoint will return a fully written, HTML-formatted article."
    title: Input the title of an article that you'd like written, and this endpoint will return a fully written, HTML-formatted article with that title.
        language: Enter the language that your keyword is in, and that you'd like your article to be generated in.
        
    """
    url = f"https://byword-article-generation.p.rapidapi.com/rapidapi"
    querystring = {'title': title, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "byword-article-generation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def write_article_from_keyword(keyword: str, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Input a keyword that you'd like to rank for on SEO/organic search, and this endpoint will return a fully written article which can rank for that keyword. A keyword can be multiple words long (e.g. 'how to get rid of cold')."
    keyword: Enter the keyword that you want to rank for, and the API will generate an article which can rank for that keyword. A keyword can contain multiple words, e.g. 'how change car engine oil'.
        language: Enter the language that your keyword is in, and that you'd like your article to be generated in.
        
    """
    url = f"https://byword-article-generation.p.rapidapi.com/rapidapi"
    querystring = {'keyword': keyword, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "byword-article-generation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

