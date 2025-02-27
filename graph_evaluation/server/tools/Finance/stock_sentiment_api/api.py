import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def news_sentiment_data(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of articles related to the input stock and their associated sentiment score.
		The sentiment score is called compound."
    
    """
    url = f"https://stock-sentiment-api.p.rapidapi.com/stock_news_sentiment/"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-sentiment-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_negative_news(ticker: str, count: str='3', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of the most negative articles related to the given stock. The sentiment scores can be viewed in the compound field and you can change the number of articles returned with the count parameter.
		
		Note: The api parses for 100 articles so any count given over 100, the api will just return all 100 articles"
    
    """
    url = f"https://stock-sentiment-api.p.rapidapi.com/top_negative_news/"
    querystring = {'ticker': ticker, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-sentiment-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_positive_news(ticker: str, count: str='3', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of the most positive articles related to the given stock. The sentiment scores can be viewed in the compound field and you can change the number of articles returned with the count parameter.
		
		Note: The api parses for 100 articles so any count given over 100, the api will just return all 100 articles"
    
    """
    url = f"https://stock-sentiment-api.p.rapidapi.com/top_positive_news/"
    querystring = {'ticker': ticker, }
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-sentiment-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_data(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of daily stock metrics such as opening, closing, highest and lowest prices."
    
    """
    url = f"https://stock-sentiment-api.p.rapidapi.com/market_data/"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-sentiment-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

