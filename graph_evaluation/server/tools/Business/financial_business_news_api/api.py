import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_article_by_uuid(uuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Get Article by UUID
		Will return an article matching a specific UUID number supplied as path parameter."
    
    """
    url = f"https://financial-business-news-api.p.rapidapi.com/api/v1/news/article/{uuid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-business-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_articles_by_exchange(exchange_code: str, page: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Get Articles By Exchange Code & Page Number
		This endpoint will return articles in a given exchange and allows 
		you to page through the articles by supplying the page number as a parameter
		for a list of exchange_code see the exchange list API"
    
    """
    url = f"https://financial-business-news-api.p.rapidapi.com/api/v1/news/articles-by-exchange/{exchange_code}"
    querystring = {'page': page, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-business-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stock_tickers_by_exchange_code(exchange_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Get Stock Tickers By Exchange Code
		will return a list of stock tickers corresponding to an exchange code, in order to obtain exchange code see list of exchanges endpoints"
    
    """
    url = f"https://financial-business-news-api.p.rapidapi.com/api/v1/news/tickers-by-exchange/{exchange_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-business-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_exchanges(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Get List Of Supported Exchanges
		Get a complete list of supported exchanges"
    
    """
    url = f"https://financial-business-news-api.p.rapidapi.com/api/v1/news/list-exchanges"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-business-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_articles_published_by_a_certain_publisher(publisher: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Get Articles Published by a certain Publisher
		
		This returns articles published by a specific publisher
		for example you can find articles published by 
		
		- Motley Fool
		- Yahoo Finance
		- Pr NewsWire
		- Investors Business Daily
		- Reuters
		- Barrons
		- TheStreet.com 
		
		and many many more"
    
    """
    url = f"https://financial-business-news-api.p.rapidapi.com/api/v1/news/articles-by-publisher/{publisher}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-business-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_publishers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Get List of Publishers
		This endpoint will return a complete list of publishers for business articles."
    
    """
    url = f"https://financial-business-news-api.p.rapidapi.com/api/v1/news/list-publishers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-business-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_articles_by_page_number(page_number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Get Articles By Page Number
		This returns articles grouped by page numbers grouped from newest articles to oldest for instance page 1 articles will contain newer articles and page 2 older articles and so on"
    
    """
    url = f"https://financial-business-news-api.p.rapidapi.com/api/v1/news/articles-by-page/{page_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-business-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_articles_by_company_ticker(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Get Articles By Company Ticker
		This will return articles associated with a company ticker symbol.
		for example to return articles related to APPLE you can use ticker symbol AAPL."
    
    """
    url = f"https://financial-business-news-api.p.rapidapi.com/api/v1/news/articles-by-ticker/{ticker_symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-business-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_articles_published_by_date(published_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Get Articles Published By Date
		This returns articles published on a specific date,"
    
    """
    url = f"https://financial-business-news-api.p.rapidapi.com/api/v1/news/articles-by-date/{published_date}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-business-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_articles_by_upper_bound(bound: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Get Articles By Upper Bound
		This returns articles from 1 to a specific limit indicated in the url path parameter.
		
		if the URL ends with 10 then ten articles will be returned and so on."
    
    """
    url = f"https://financial-business-news-api.p.rapidapi.com/api/v1/news/articles-bounded/{bound}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-business-news-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

