import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_individual_news_source_news(newspaperid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news about Covid-19 updates from a specific news source.
		
		/news/sydneymorningherald - Sydney Morning Herald(Australia)
		/news/timesofindia - The Times of India(India)
		/news/telegraph - The Telegraph(UK)
		/news/guardian - The Guardian(UK)
		/news/independent - Independent(UK)
		/news/dailysun - Daily Sun(Bangladesh)
		/news/nytimes - New York Times(USA)
		/news/wsj - The Wall Street Journal(USA)"
    
    """
    url = f"https://covid-19-news4.p.rapidapi.com/news/{newspaperid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-news4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_coronavirus_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will return news about about Covid-19 updates from all over the world."
    
    """
    url = f"https://covid-19-news4.p.rapidapi.com/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "covid-19-news4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

