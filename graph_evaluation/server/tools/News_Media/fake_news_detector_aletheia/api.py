import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def scrape_and_detect_fake_news_url(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Scrape a news website by passing URL as a parameter and receive the article that is scraped as well as the Algorithm Model Detection."
    
    """
    url = f"https://fake-news-detector-aletheia.p.rapidapi.com/scraper"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-news-detector-aletheia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def detect_fake_news_text(news: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Insert text-based news in the 'news' parameter to fact check using machine learning"
    
    """
    url = f"https://fake-news-detector-aletheia.p.rapidapi.com/"
    querystring = {'news': news, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fake-news-detector-aletheia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

