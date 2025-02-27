import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def news_sentiment(ticker: str, enddate: str, startdate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get News Sentiment and News volume for a particular stock.
		sentiment>0 indicates positive sentiment. sentiment<0 indicates negative sentiment."
    
    """
    url = f"https://stock-news-sentiment-stockshark.p.rapidapi.com/getNewsSentiment"
    querystring = {'ticker': ticker, 'endDate': enddate, 'startDate': startdate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-news-sentiment-stockshark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

