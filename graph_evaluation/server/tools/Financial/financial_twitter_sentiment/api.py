import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def financial_twitter_sentiment(stocks: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Real-time financial Twitter sentiment of a stock (e.g. $aapl).
		
		Our API collects the latest tweets where the stock(s) are mentioned and based on the tweet text our algorithm is able to evaluate the sentiment of the tweet for the corresponding stock(s)
		
		This API accepts the maximum of 5 stocks per call.
		The query parameter is cashtag+stock_ticker and each stock separated by a comma.
		
		The response is composed by:
		
		- date: Current server time and date
		
		- name: Name of the stock ticker searched
		
		- sentiment: Sentiment of the stock(s) search. A sentiment value higher than 1 corresponds to a positive sentiment. A value between 0 and 1 is a neutral sentiment. A negative sentiment corresponds to a negative sentiment."
    
    """
    url = f"https://financial-twitter-sentiment.p.rapidapi.com/api/fin-twitter/stocks/sentiment"
    querystring = {'stocks': stocks, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-twitter-sentiment.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

