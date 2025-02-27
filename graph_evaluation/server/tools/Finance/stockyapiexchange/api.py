import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_7_latest(keyword: str='GOOG', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This feature provides the most recent information about a stock, including its current price, percentage change, and trading volume. It's updated in real-time as new data becomes available."
    
    """
    url = f"https://stockyapiexchange.p.rapidapi.com/latest"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockyapiexchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_6_daily(startdate: str='2023-02-01', keyword: str='GOOG', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This feature provides up-to-date information about a stock's performance during a trading day. It includes the stock's opening and closing prices, as well as its high, low, and trading volume for the day."
    
    """
    url = f"https://stockyapiexchange.p.rapidapi.com/daily"
    querystring = {}
    if startdate:
        querystring['startDate'] = startdate
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockyapiexchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_5_history(startdate: str='2023-02-01', keyword: str='GOOG', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This feature provides historical data for a stock, including its open, high, low, close, and trading volume for each day. Users can access data for different time periods, such as daily, weekly, monthly, or yearly."
    
    """
    url = f"https://stockyapiexchange.p.rapidapi.com/history"
    querystring = {}
    if startdate:
        querystring['startDate'] = startdate
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockyapiexchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_4_charts(keyword: str='GOOG', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This feature provides interactive charts that allow users to visualize the performance of a stock over time. Users can customize the time period and type of chart (line, bar, candlestick, etc.) to get a better understanding of the stock's price movements."
    
    """
    url = f"https://stockyapiexchange.p.rapidapi.com/charts"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockyapiexchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_3_summary(keyword: str='GOOG', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Summary: This feature gives a quick overview of the performance of a stock, including its current price, percentage change, and trading volume. It also provides a summary of the key financial metrics and news related to the stock"
    
    """
    url = f"https://stockyapiexchange.p.rapidapi.com/summary"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockyapiexchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_2_stockdetails(keyword: str='GOOG', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stockdetails: This feature provides detailed information about a specific stock, including its name, ticker symbol, market capitalization, sector, industry, and other key financial metrics such as price-to-earnings ratio (P/E ratio), dividend yield, and earnings per share (EPS)"
    
    """
    url = f"https://stockyapiexchange.p.rapidapi.com/stockdetails"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockyapiexchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_autocomplete(keyword: str='GOOG', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "AUTOComplete: This feature allows users to easily search for stocks by providing suggested completions for a ticker symbol or company name as they type. It helps users find the right stock without having to know the exact spelling or symbol."
    
    """
    url = f"https://stockyapiexchange.p.rapidapi.com/autocomplete"
    querystring = {}
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stockyapiexchange.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

