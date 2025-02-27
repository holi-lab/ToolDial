import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_tesla_news_from_top_financial_news_sites(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It refers to gathering news articles and updates on the electric vehicle company, Tesla, from reputable and reliable financial news sources. These sources provide in-depth coverage of Tesla's financial performance, operational developments, strategic moves, and impact on the automotive industry. By collecting news from top financial news sites, one can stay informed about the latest updates and expert analysis on the company, and gain a comprehensive understanding of Tesla's current situation and future prospects."
    
    """
    url = f"https://tesla_news_api.p.rapidapi.com/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tesla_news_api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

