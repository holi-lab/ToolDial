import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def summarize(text: str, max_sentences: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Summarizes an article by fetching text from a specified URL or reading the input text and generating a summary for a web article."
    text: This field contains the text input for the text summarization request. It can be either a URL or the raw text of an article.
        
    """
    url = f"https://tldr-text-analysis.p.rapidapi.com/summarize/"
    querystring = {'text': text, 'max_sentences': max_sentences, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tldr-text-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sentiment_analysis(text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Performs sentiment analysis on a web article or text input."
    text: This field contains the text input for the sentiment analysis request. It can be either a URL or the raw text of an article.
        
    """
    url = f"https://tldr-text-analysis.p.rapidapi.com/sentiment_analysis/"
    querystring = {'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tldr-text-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def extract_keywords(n_keywords: int, text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extracts keywords from an article given the URL for the article and the number of keywords to search for."
    text: This field contains the text input for the keyword extraction request. It can be either a URL or the raw text of an article.
        
    """
    url = f"https://tldr-text-analysis.p.rapidapi.com/keywords/"
    querystring = {'n_keywords': n_keywords, 'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tldr-text-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

