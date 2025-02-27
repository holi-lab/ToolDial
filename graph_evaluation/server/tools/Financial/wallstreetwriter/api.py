import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_financial_article(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create ready to use financial articles. Simply input a stock ticker and receive a short article incorporating recent stock price data and relevant news publications."
    
    """
    url = f"https://wallstreetwriter.p.rapidapi.com/generate"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wallstreetwriter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

