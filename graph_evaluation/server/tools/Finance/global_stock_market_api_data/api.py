import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_most_popular_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Featured here: links to the most widely read news articles from all sections as determined by readers."
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/news/most_popular_news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_52_week_low_by_fundamental_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 52 week low by Fundamental with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/week_low_countrywise_by_fundamental/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_active_stocks_by_fundamental_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Most Active Stocks by Fundamental with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/most_active_countrywise_by_fundamental/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_active_stocks_by_technical_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Most Active Stocks by Technical with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/most_active_countrywise_by_technical/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_active_stocks_by_performance_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Most Active Stocks by performance with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/most_active_countrywise_by_performance/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_trending_stocks_by_technical_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Trending Stocks by Technical with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/stocks_countrywise_by_technical/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_trending_stocks_by_fundamental_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Trending Stocks by Fundamental with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/stocks_countrywise_by_fundamental/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_trending_stocks_by_performance_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Trending Stocks by performance with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/stocks_countrywise_by_performance/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def active_stocks_by_price_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Active Stocks by price with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/most_active_countrywise_by_price/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_52_week_low_by_technical_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 52 Week Low by Technical with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/week_low_countrywise_by_technical/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_52_weel_low_by_performance_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 52 Weel Low by performance with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/week_low_countrywise_by_performance/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_52_week_low_by_price_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 52 Week Low by price with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/week_low_countrywise_by_price/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_52_week_high_by_fundamental_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 52 week high by Fundamental with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/week_high_countrywise_by_fundamental/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_52_week_high_by_technical_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 52 Week High by Technical with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/week_high_countrywise_by_technical/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_52_weel_high_by_performance_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 52 Weel High by performance with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/week_high_countrywise_by_performance/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_52_week_high_by_price_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get 52 Week High by price with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/week_high_countrywise_by_price/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_trending_stocks_by_price_with_respecto_to_country(countrycode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Trending Stocks by price with respecto to Country"
    countrycode: Please use the **Value** key pair that is returned from /countryList API
        
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/stocks_countrywise_by_price/{countrycode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_country_of_names(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API will return the list of **ALL Country** names, please use the value key pair and get the trending stocks, week 52 high, low, top gainer and top loser respective to Country"
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/countryList"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_economy_indicators_news(pageno: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Keep track of breaking economic statistics and developments. Stay up-to-date on key metrics and economic indicators such as unemployment, housing, GDP and more."
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/news/economy_indicators_news/{pageno}"
    querystring = {}
    if pageno:
        querystring['pageNo'] = pageno
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_economy_news(pageno: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Breaking economic news, with reports on global stock markets, personal finance, businesses and technology."
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/news/economy_news/{pageno}"
    querystring = {}
    if pageno:
        querystring['pageNo'] = pageno
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_forex_news(pageno: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Keep track of the latest currency market moves as they develop. Stay up-to-date on breaking forex news as well as relevant financial developments. Use our Forex tips."
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/news/forex_news/{pageno}"
    querystring = {}
    if pageno:
        querystring['pageNo'] = pageno
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_commodities_news(pageno: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The latest commodity market moves as well as developments in the futures markets. Stay up-to-date with breaking commodities news and specifics on major commodities such as crude oil and gold."
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/news/commodities_news/{pageno}"
    querystring = {}
    if pageno:
        querystring['pageNo'] = pageno
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stock_market_news(pageno: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stock market news from the US and around the world. Our news team reports on market moving events around the world, that traders need to know in order to stay on top of developments in the companies whose stocks they trade."
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/news/stock_market_news/{pageno}"
    querystring = {}
    if pageno:
        querystring['pageNo'] = pageno
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cryptocurrency_news(pageno: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Investing.com's crypto news section is providing the latest cryptocurrency news - stay up-to-date with breaking news on major cryptocurrencies, including Bitcoin, Ethereum, Litecoin, Ripple and ICOs."
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/news/cryptocurrency_news/{pageno}"
    querystring = {}
    if pageno:
        querystring['pageNo'] = pageno
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View the latest financial news articles published on Investing.com. Read the most recent stories and breaking news, covering all aspects of financial markets worldwide, including commodities, stocks, currencies, indices and more."
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/news/latest_news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_major_commodities_by_technical(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "World financial market to get major commodities with respect to technical"
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/major_commodity_by_technical"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_major_commodities_by_performance(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "World financial market to get major commodities with respect to performance"
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/major_commodity_by_performance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_major_commodities_by_price(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "World financial market to get major commodities with respect to price"
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/major_commodity_by_price"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_global_indices_by_technical(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "World financial market to get global Indices with respect to technical"
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/global_indices_by_technical"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_global_indices_by_performance(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "World financial market to get global Indices with respect to performance"
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/global_indices_by_performance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_global_indices_by_price(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "World financial market to get global Indices with respect to price"
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/global_indices_by_price"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_major_global_indices_by_technical(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "World financial market to get major global Indices with respect to technical"
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/major_global_indices_by_technical"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_major_global_indices_by_performance(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "World financial market to get major global Indices with respect to performance"
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/major_global_indices_by_performance"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_major_global_indices_by_price(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "World financial market to get major global Indices with respect to price"
    
    """
    url = f"https://global-stock-market-api-data.p.rapidapi.com/major_global_indices_by_price"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-stock-market-api-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

