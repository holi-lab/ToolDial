import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_historical_company_article_data(ticker: str, months: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides historical data for a given company including news, polarity, sentiment, and more.
		
		`{10 items
		"FeedUrl":"https://finance.yahoo.com/news/electronic-arts-shoots-higher-amazon-073805755.html?.tsrc=rss"
		"Polarity":0
		"PublishDate":"2022-08-26T07:38:05+00:00"
		"Subjectivity":1.5
		"Title":"Electronic Arts Shoots Higher on Amazon Bid Report"
		"VaderAggregate":0.5
		"VaderNegative":0.2
		"VaderNeutral":0.5
		"VaderPositive":0.8
		"id":"67988e33-4ded-3b46-b2c6-9f2d96580132"
		}`"
    
    """
    url = f"https://stoxx1.p.rapidapi.com/api/v1/stoxx/company/{ticker}/history/articles/{months}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stoxx1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_publicly_traded_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a list of all publicly traded companies"
    
    """
    url = f"https://stoxx1.p.rapidapi.com/api/v1/stoxx/companies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stoxx1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_historical_data(ticker: str, months: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides access to daily historical company pricing data over a given period of months
		
		A sample output for each day:
		`{
		"changePercent":0
		"close":115.1465
		"date":"2022-05-27"
		"high":115.187
		"low":112.628
		"open":113.55
		"symbol":"AMZN"
		"updated":1654726813000
		"volume":93660160
		}`"
    
    """
    url = f"https://stoxx1.p.rapidapi.com/api/v1/stoxx/company/{ticker}/history/prices/{months}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stoxx1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_quote_price_data(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve updated quote and pricing data for a given company. This endpoint provides you with:
		
		- Symbol
		- Company Name
		- Exchange
		- High
		- Low
		- Close
		- PE
		- Volume
		- Market Cap
		- Datetime"
    
    """
    url = f"https://stoxx1.p.rapidapi.com/api/v1/stoxx/company/{ticker}/quote"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stoxx1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_market_performance(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the overall current market performance
		
		```
		{
		"datetimeUpdated":1661544000005
		"performance":-0.01535
		"sector":"Utilities"
		"symbol":"XLU"
		}
		```"
    
    """
    url = f"https://stoxx1.p.rapidapi.com/api/v1/stoxx/market/performance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stoxx1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_competition(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a company's known competitors"
    
    """
    url = f"https://stoxx1.p.rapidapi.com/api/v1/stoxx/company/{ticker}/competition"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stoxx1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_company_information(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides detailed information for a given company:
		- Symbol
		- Company Name
		- Industry
		- Website
		- Relevant Tags
		- General Sector
		- Country
		- Zip Code"
    
    """
    url = f"https://stoxx1.p.rapidapi.com/api/v1/stoxx/company/{ticker}/info"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stoxx1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

