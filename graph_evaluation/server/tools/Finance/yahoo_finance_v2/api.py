import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stock_history(interval: str, symbol: str, diffandsplits: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Historic data for stocks, ETFs, mutuals funds, etc..."
    interval: Allows one of following : 5m|15m|30m|1h|1d|1wk|1mo|3mo
        symbol: A single symbol
        diffandsplits: Allows one of following : true|false
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/hi/history/{symbol}/{interval}"
    querystring = {}
    if diffandsplits:
        querystring['diffandsplits'] = diffandsplits
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_quotes_stocks(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Quote data for stocks, ETFs, mutuals funds, etc..."
    symbol: Multiple symbols separated by commas. Max is 200
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def options(symbol: str, expiration: int=1705622400, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get option data for stocks, ETFs and indexes."
    symbol: A single stock symbol 
        expiration: Expiration date
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/op/option/{symbol}"
    querystring = {}
    if expiration:
        querystring['expiration'] = expiration
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_financial_data(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock financial data."
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/financial-data"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Recently published stock news in all sectors."
    
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/ne/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insider_trades(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Latest insider trading activities from CEO, Directors, Chief Executive Officer, 10% Owner, etc..."
    
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/v1/sec/form4"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_news_stock(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Recently published stock news in Yahoo finance."
    symbol: A single stock symbol 
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/ne/news/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_combine_data(symbol: str, module: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get combine stock data such as profile, financial data, statistics, balance sheet, sec-filing, quote, earnings, trends and more!"
    symbol: A single symbol.
        module: `asset-profile`, `income-statement`, `balance-sheet,` `cashflow-statement`,  `default-key-statistics`, `calendar-events`, `sec-filings`, `upgrade-downgrade-history`, `institution-ownership`, `fund-ownership`, `insider-transactions`, `insider-holders`, `earnings-history`
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/mo/module/{symbol}"
    querystring = {'module': module, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_aggressive_small_caps(start: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Small cap stocks with earnings growth rates better than 25%."
    start: Enter a start index
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/co/collections/aggressive_small_caps"
    querystring = {}
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_undervalued_large_caps(start: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Large cap stocks that are potentially undervalued."
    start: Enter a start index
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/co/collections/undervalued_large_caps"
    querystring = {}
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_most_actives(start: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stocks ordered in descending order by intraday trade volume."
    start: Enter a start index
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/co/collections/most_actives"
    querystring = {}
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_small_cap_gainers(start: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Small Caps with a 1 day price change of 5.0% or more."
    start: Enter a start index
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/co/collections/small_cap_gainers"
    querystring = {}
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_most_watched(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Trending stocks in today's market"
    
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/tr/trending"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_day_losers(start: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stocks ordered in ascending order by price percent change with respect to the previous close."
    start: Enter a start index
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/co/collections/day_losers"
    querystring = {}
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_day_gainers(start: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stocks ordered in descending order by price percent change with respect to the previous close."
    start: Enter a start index
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/co/collections/day_gainers"
    querystring = {}
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_undervalued_growth_stocks(start: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stocks with earnings growth rates better than 25% and relatively low PE and PEG ratios."
    start: Enter a start index
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/co/collections/undervalued_growth_stocks"
    querystring = {}
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_top_gainers_depreciated(start: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stocks with the most gains in today's market"
    start: Enter a start index
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/ga/topgainers"
    querystring = {}
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_growth_technology_stocks(start: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Technology stocks with revenue and earnings growth in excess of 25%."
    start: Enter a start index
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/co/collections/growth_technology_stocks"
    querystring = {}
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_key_statistics(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock key statistics data."
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/default-key-statistics"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_insider_holders(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock insider holders' information."
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/insider-holders"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_earnings_history(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get earnings history information for a particular stock"
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/earnings-history"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_upgrade_downgrade_history(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock upgrade and downgrade history."
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/upgrade-downgrade-history"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_insider_transactions(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock insider transactions history."
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/insider-transactions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_income_statement(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock income statement data."
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/income-statement"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_index_trend(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get index trend earnings history information for a particular stock"
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/index-trend"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_cashflow_statement(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock cash flow statements."
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/cashflow-statement"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_profile(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock profile information such as company name, descriptions, website, etc..."
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/asset-profile"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_calendar_events(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock calendar events."
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/calendar-events"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_balance_sheet(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock balance sheet data."
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/balance-sheet"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_recommendation_trend(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock recommendations and trends."
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/recommendation-trend"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_net_share_purchase_activity(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get net share purchase activity information for a particular stock"
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/net-share-purchase-activity"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_institution_ownership(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock institution ownership."
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/institution-ownership"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_sec_filings(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock SEC filings."
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/sec-filings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_earnings(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get earnings information for a particular stock"
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/earnings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_earnings_trend(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get earnings trend earnings history information for a particular stock"
    symbol: A single symbol
        
    """
    url = f"https://yahoo-finance15.p.rapidapi.com/api/yahoo/qu/quote/{symbol}/earnings-trend"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yahoo-finance15.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

