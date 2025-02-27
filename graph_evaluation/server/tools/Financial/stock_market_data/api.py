import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def wsb_daily_mentions(ticker_symbol: str, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get r/WallStreetBets posts and their respective sentiment scores for the date and stock ticker mentioned."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/buzz/wsb/daily-mentions"
    querystring = {'ticker_symbol': ticker_symbol, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def google_trends(ticker_symbol: str, timeframe: str='1-d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch Google Trends data for a given stock and timeframe."
    timeframe: You may enter the timeframe for the trends data required:
ex.  now 4-H, now 1-H, now 1-d, now 7-d, today 1-m, today 3-m, today 5-y. The data fetched by default is for a 1-year timeframe
        
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/buzz/google-trends"
    querystring = {'ticker_symbol': ticker_symbol, }
    if timeframe:
        querystring['timeframe'] = timeframe
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rate_of_change(period: str='5', ticker_symbol: str='NFLX', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "calculate rate of change for the ticker and periods input"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/indicator/roc"
    querystring = {}
    if period:
        querystring['period'] = period
    if ticker_symbol:
        querystring['ticker_symbol'] = ticker_symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def s_p_600(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all stocks in the S&P 600 index. See 
		https://en.wikipedia.org/wiki/List_of_S%26P_600_companies for details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/s-and-p-six-hundred"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wsb_historical_statistics(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch day-wise mention data of the given stock ticker on r/WallStreetBets."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/buzz/wsb/historical-stats"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def risk_free_rate(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The current risk free rate of return. See this article for more info on what this means: https://www.investopedia.com/terms/r/risk-freerate.asp"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/economy/risk-free-rate"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def last_year_market_return(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The market's return in the last 365 days, tracked via the S&P 500 index."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/economy/last-year-market-return"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def key_stats(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting the key statistic"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/key-stats"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quote(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current stock price quote data."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/quote"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_info(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets company info"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/company-info"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_valuation_measures(ticker_symbol: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A table of the company's historical valuation measures, served in CSV format."
    format: The format the time series data will be returned in. Acceptable values are \"json\" and \"csv\".
        
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/valuation/historical-valuation-measures"
    querystring = {'ticker_symbol': ticker_symbol, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def valuation_measures(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The company's key valuation measures."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/valuation/valuation-measures"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cost_of_equity(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The stock's theoretically cost of equity, based on the CAPM model as specified here: https://www.investopedia.com/terms/c/capm.asp"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/valuation/cost-of-equity"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def enterprise_value(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The stock's enterprise value, as defined here: https://www.investopedia.com/terms/e/enterprisevalue.asp"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/valuation/enterprise-value"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_quarterly_income_statement(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The company's latest quarterly income statement."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/financials/income-statement/quarterly-latest"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_annual_income_statement(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The company's latest annual income statement."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/financials/income-statement/annual-latest"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_annual_balance_sheet(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The company's current annual balance sheet."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/financials/balance-sheet/annual-latest"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_quarterly_balance_sheet(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The company's current quarterly balance sheet statement."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/financials/balance-sheet/quarterly-latest"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_price_correlation(ticker_symbol1: str, ticker_symbol2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Pearson correlation coefficient between the prices of two stocks."
    ticker_symbol1: The ticker symbol of the first stock to compare.
        ticker_symbol2: The ticker symbol of the second stock to compare.
        
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock-similarity/stock-price-correlation"
    querystring = {'ticker_symbol1': ticker_symbol1, 'ticker_symbol2': ticker_symbol2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def industry_similarity(ticker_symbol2: str, ticker_symbol1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The industry similarity score between two stocks, reported as a value between 0 and 2."
    ticker_symbol2: The ticker symbol of the second stock to compare.
        ticker_symbol1: The ticker symbol of the first stock to compare.
        
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock-similarity/industry-similarity"
    querystring = {'ticker_symbol2': ticker_symbol2, 'ticker_symbol1': ticker_symbol1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def business_description_similarity(ticker_symbol1: str, ticker_symbol2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The natural language similarity between the textual descriptions of the two businesses. Behind the scenes, this endpoint computes the cosine similarity between the TF-IDF vectors associated with the descriptions of the two businesses."
    ticker_symbol1: The ticker symbol of the first stock to compare.
        ticker_symbol2: The ticker symbol of the second stock to compare.
        
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock-similarity/business-description-similarity"
    querystring = {'ticker_symbol1': ticker_symbol1, 'ticker_symbol2': ticker_symbol2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quarterly_balance_sheets(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw quarterly balance sheet data directly from Yahoo Finance."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/quarterly-balance-sheets"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def annual_balance_sheets(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw annual balance sheet data directly from Yahoo Finance."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/annual-balance-sheets"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quarterly_income_statements(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw quarterly income statement data directly from Yahoo Finance."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/quarterly-income-statements"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def annual_income_statements(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw annual income statement data from Yahoo Finance."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/annual-income-statements"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calendar_events(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw calendar events data from Yahoo Finance."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/calendar-events"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def esg_scores(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw ESG score data from Yahoo Finance."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/esg-scores"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upgrade_downgrade_history(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw analyst upgrade downgrade history data directly from Yahoo Finance."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/upgrade-downgrade-history"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def financial_data(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw stock financial data from Yahoo Finance."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/financial-data"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earnings(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw earnings data from Yahoo Finance."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/earnings"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_profile(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw company profile data from Yahoo Finance."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/company-profile"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def key_statistics(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw key stock statistics data, directly from Yahoo Finance."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/stats"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def summary_detail(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw company summary detail from Yahoo Finance."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/summary-detail"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def price(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw price data from Yahoo Finance."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/price"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_prices(ticker_symbol: str, format: str='json', years: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets historical prices, in CSV format."
    format: The format the time series data will be returned in. Acceptable values are 'json' and 'csv'.
        years: The number of years of historical price data to retrieve.
        
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/historical-prices"
    querystring = {'ticker_symbol': ticker_symbol, }
    if format:
        querystring['format'] = format
    if years:
        querystring['years'] = years
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def annual_historical_balance_sheets(ticker_symbol: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Historical annual balance sheets, served in CSV format."
    format: The format the time series data will be returned in. Acceptable values are 'json' and 'csv'.
        
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/financials/balance-sheet/annual-historical"
    querystring = {'ticker_symbol': ticker_symbol, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def annual_historical_income_statements(ticker_symbol: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Historical annual income statements, served in CSV format."
    format: The format the time series data will be returned in. Acceptable values are 'json' and 'csv'.
        
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/financials/income-statement/annual-historical"
    querystring = {'ticker_symbol': ticker_symbol, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quarterly_historical_balance_sheets(ticker_symbol: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Historical quarterly balance sheets, served in CSV format."
    format: The format the time series data will be returned in. Acceptable values are 'json' and 'csv'.
        
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/financials/balance-sheet/quarterly-historical"
    querystring = {'ticker_symbol': ticker_symbol, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quarterly_historical_income_statements(ticker_symbol: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Historical quarterly income statements, served in CSV format."
    format: The format the time series data will be returned in. Acceptable values are 'json' and 'csv'.
        
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/financials/income-statement/quarterly-historical"
    querystring = {'ticker_symbol': ticker_symbol, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hedge_basket(hedged_ticker_symbol: str, basket_ticker_symbols: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides an easy way to hedge a stock using a basket of other stocks which collectively behave as a hedge against the desired stock."
    hedged_ticker_symbol: The ticker symbol of the stock for which to create a hedging basket.
        basket_ticker_symbols: The list of ticker symbols to consider including in the hedging basket.
        
    """
    url = f"https://stock-market-data.p.rapidapi.com/algorithms/hedge-basket"
    querystring = {'hedged_ticker_symbol': hedged_ticker_symbol, 'basket_ticker_symbols': basket_ticker_symbols, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def london_stock_exchange(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All stocks traded on the LondonStockExchange."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/exchange/london-stock-exchange"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def options_contracts(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw options contracts data for the given stock from Yahoo Finance."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/options-contracts"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def raw_quote(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw quote data from Yahoo Finance."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/quote"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def raw_historical_prices(ticker_symbol: str, format: str='json', years: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw historical prices data in CSV format from Yahoo Finance."
    format: The format the time series data will be returned in. Acceptable values are 'json' and 'csv'.
        years: The number of years of historical price data to retrieve.
        
    """
    url = f"https://stock-market-data.p.rapidapi.com/yfinance/historical-prices"
    querystring = {'ticker_symbol': ticker_symbol, }
    if format:
        querystring['format'] = format
    if years:
        querystring['years'] = years
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_actives(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stocks listed on https://finance.yahoo.com/screener/predefined/most_actives"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/screener/most-actives"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conservative_foreign_funds(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Funds listed on https://finance.yahoo.com/screener/predefined/conservative_foreign_funds"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/screener/conservative-foreign-funds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def high_yield_bonds(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bonds listed on https://finance.yahoo.com/screener/predefined/high_yield_bond"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/screener/high-yield-bonds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aggressive_small_caps(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Aggressive small caps https://finance.yahoo.com/screener/predefined/aggressive_small_caps"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/screener/aggressive-small-caps"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def undervalued_large_caps(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stocks listed on https://finance.yahoo.com/screener/predefined/undervalued_large_caps"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/screener/undervalued-large-caps"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def small_cap_gainers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stocks listed on https://finance.yahoo.com/screener/predefined/small_cap_gainers"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/screener/small-cap-gainers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_mutual_funds(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mutual funds listed on https://finance.yahoo.com/screener/predefined/top_mutual_funds"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/screener/top-mutual-funds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def portfolio_anchors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Funds listed on https://finance.yahoo.com/screener/predefined/portfolio_anchors"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/screener/portfolio-anchors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def solid_large_growth_funds(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Funds listed on https://finance.yahoo.com/screener/predefined/solid_large_growth_funds"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/screener/solid-large-growth-funds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def solid_midcap_growth_funds(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Funds listed on https://finance.yahoo.com/screener/predefined/solid_midcap_growth_funds"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/screener/solid-midcap-growth-funds"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nikkei_225(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks comprising the Nikkei 225 index. See https://en.wikipedia.org/wiki/Nikkei_225 for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/nikkei-two-twenty-five"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def omx_nordic_40(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks comprising the OMX Nordic 40 index. See https://en.wikipedia.org/wiki/OMX_Nordic_40 for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/omx-nordic-forty"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def undervalued_growth_stocks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stocks listed on https://finance.yahoo.com/screener/predefined/undervalued_growth_stocks"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/screener/undervalued-growth-stocks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nyse_arca_major_market_index(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks comprising the NYSE Arca Major Market Index. See https://en.wikipedia.org/wiki/NYSE_Arca_Major_Market_Index for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/nyse-arca-major-market-index"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def day_losers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stocks listed on https://finance.yahoo.com/screener/predefined/day_losers"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/screener/day-losers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def day_gainers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stocks listed on https://finance.yahoo.com/screener/predefined/day_gainers"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/screener/day-gainers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def growth_technology_stocks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stocks listed on https://finance.yahoo.com/screener/predefined/growth_technology_stocks"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/screener/growth-technology-stocks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dow_jones_global_titans_50(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks comprising the Dow Jones Global Titans 50 index. See https://en.wikipedia.org/wiki/Dow_Jones_Global_Titans_50 for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/dow-jones-global-titans-fifty"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dax_30(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks comprising the DAX 30 index. See https://en.wikipedia.org/wiki/DAX for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/dax-thirty"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def euro_next_100(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks comprising the Euro Next 100 index. See https://www.dividendmax.com/market-index-constituents/euronext-100 for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/euro-next-one-hundred"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def down_jones_transportation_average(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks comprising the Down Jones Transportation Average. See https://en.wikipedia.org/wiki/Dow_Jones_Transportation_Average for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/down-jones-transportation-average"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def phlx_semiconductor_sector(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks comprising the PHLX Semiconductor Sector index. See https://en.wikipedia.org/wiki/PHLX_Semiconductor_Sector for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/phlx-semiconductor"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def down_jones_utility_average(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks comprising the Down Jones Utility Average index. See https://en.wikipedia.org/wiki/Dow_Jones_Utility_Average for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/down-jones-utility-average"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nasdaq_100(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks comprising the Nasdaq 100 index. See https://en.wikipedia.org/wiki/NASDAQ-100 for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/nasdaq-one-hundred"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def phlx_gold_silver_index(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks comprising the PHLX Gold & Silver Index index. See https://en.wikipedia.org/wiki/Philadelphia_Gold_and_Silver_Index for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/phlx-gold-and-silver"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fear_greed_index(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "From CNN Money: https://money.cnn.com/data/fear-and-greed/"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/buzz/fear-and-greed-index"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shanghai_stock_exchange(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of stocks traded on the Shanghai Stock Exchange."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/exchange/shanghai-stock-exchange"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hong_kong_stock_exchange(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks traded on the Hong Kong Stock Exchange."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/exchange/hong-kong-stock-exchange"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_public_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All ticker symbols traded on the major American stock exchanges."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/all-public-companies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_york_stock_exchange(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All ticker symbols traded on the New York Stock Exchange."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/exchange/nyse"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nasdaq(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All ticker symbols traded on the Nasdaq Stock Exchange."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/exchange/nasdaq"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def s_p_500(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all stocks in the S&P 500 index. See https://en.wikipedia.org/wiki/S%26P_500 for details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/s-and-p-five-hundred"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nasdaq_composite(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all stocks in the Nasdaq Composite index. See https://en.wikipedia.org/wiki/Nasdaq_Composite for details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/nasdaq-composite"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def russel_1000(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all stocks in the Russel 1000 index. See https://www.investopedia.com/terms/r/russell_1000index.asp for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/russel-one-thousand"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nifty_50(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of stocks in the NIFTY 50 index. See https://en.wikipedia.org/wiki/NIFTY_50 for details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/nifty-fifty"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ftse_250(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of stocks in the FTSE 250 index. See https://en.wikipedia.org/wiki/FTSE_250_Index for details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/ftse-two-fifty"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ftse_100(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The composition of the FTSE 100 index. See https://en.wikipedia.org/wiki/FTSE_100_Index for details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/ftse-one-hundred"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def down_jones_industrial_average(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks in the Dow Jones Industrial Average. See https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average for details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/djia"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_name_ticker_symbol(company_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for the ticker symbol associated with the given company name."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/search/company-name-to-ticker-symbol"
    querystring = {'company_name': company_name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def isin_ticker_symbol(isin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for the ticker symbol associated with the given ISIN."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/search/isin-to-ticker-symbol"
    querystring = {'isin': isin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wikipedia_url_ticker_symbol(wikipedia_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a ticker symbol based on the company's wikipedia page url."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/search/wikpedia-url-to-ticker-symbol"
    querystring = {'wikipedia_url': wikipedia_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def twitter_sentiment(ticker_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The latest sentiment associated with the company on Twitter. This is a high latency endpoint (because the process of fetching and analyzing Tweets related to the stock takes a few seconds)."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/buzz/twitter-sentiment"
    querystring = {'ticker_symbol': ticker_symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_sentiment(ticker_symbol: str, date: str='2021-3-1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The latest sentiment associated with the company in recent financial news. This is a high latency endpoint (because the process of crawling and analyzing news related to the stock takes a few seconds)."
    date: The date for which to gather news and sentiment, in 'YYYY-MM-DD' format. If omitted, the endpoint defaults to the current day.
        
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/buzz/news-sentiment"
    querystring = {'ticker_symbol': ticker_symbol, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news(ticker_symbol: str, date: str='2021-3-1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The latest headlines about the company."
    date: The date for which to gather news, in 'YYYY-MM-DD' format. If omitted, the endpoint defaults to the current day.
        
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/buzz/news"
    querystring = {'ticker_symbol': ticker_symbol, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def most_shorted_stocks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of the day's most shorted stocks, from https://finance.yahoo.com/screener/predefined/most_shorted_stocks"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/screener/most-shorted-stocks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def s_p_400(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all stocks in the S&P 500 index. See https://en.wikipedia.org/wiki/S%26P_400 for details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/s-and-p-four-hundred"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def s_p_100(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all stocks in the S&P 100 index. See https://en.wikipedia.org/wiki/S%26P_100 for details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/s-and-p-one-hundred"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def russel_2000(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all stocks in the Russel 2000 index. See https://www.investopedia.com/terms/r/russell2000.asp for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/russel-two-thousand"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def niftybank(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of stocks in the NIFTYBANK index. See https://www.moneycontrol.com/indian-indices/nifty-bank-23.html for details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/niftybank"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def s_p_global_100(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all stocks in the S&P Global 100 index. See https://en.wikipedia.org/wiki/S%26P_Global_100 for details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/s-and-p-global-one-hundred"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def average_directional_index(short_period: str='5', ticker_symbol: str='NFLX', long_period: str='15', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "calculate the average directional index for the ticker and periods input"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/indicator/adx"
    querystring = {}
    if short_period:
        querystring['short_period'] = short_period
    if ticker_symbol:
        querystring['ticker_symbol'] = ticker_symbol
    if long_period:
        querystring['long_period'] = long_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def simple_moving_average(long_period: str='15', short_period: str='5', ticker_symbol: str='NFLX', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "calculate a simple moving average for the ticker and periods input"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/indicator/sma"
    querystring = {}
    if long_period:
        querystring['long_period'] = long_period
    if short_period:
        querystring['short_period'] = short_period
    if ticker_symbol:
        querystring['ticker_symbol'] = ticker_symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def moving_average_convergence_divergence(short_period: str='5', long_period: str='15', ticker_symbol: str='NFLX', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "calculate moving average convergence divergence for the ticker and periods input"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/indicator/macd"
    querystring = {}
    if short_period:
        querystring['short_period'] = short_period
    if long_period:
        querystring['long_period'] = long_period
    if ticker_symbol:
        querystring['ticker_symbol'] = ticker_symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def average_true_range(long_period: str='15', short_period: str='5', ticker_symbol: str='NFLX', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "calculate average true range for the ticker and periods input"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/indicator/atr"
    querystring = {}
    if long_period:
        querystring['long_period'] = long_period
    if short_period:
        querystring['short_period'] = short_period
    if ticker_symbol:
        querystring['ticker_symbol'] = ticker_symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def relative_strength_index(long_period: str='15', short_period: str='5', ticker_symbol: str='NFLX', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "calculate relative strength index for the ticker and periods input"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/indicator/rsi"
    querystring = {}
    if long_period:
        querystring['long_period'] = long_period
    if short_period:
        querystring['short_period'] = short_period
    if ticker_symbol:
        querystring['ticker_symbol'] = ticker_symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stochastic_oscillator(ticker_symbol: str='NFLX', short_period: str='5', long_period: str='15', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "calculate stochastic oscillator for the ticker and periods input"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/indicator/so"
    querystring = {}
    if ticker_symbol:
        querystring['ticker_symbol'] = ticker_symbol
    if short_period:
        querystring['short_period'] = short_period
    if long_period:
        querystring['long_period'] = long_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def simple_moving_average_volume(ticker_symbol: str='NFLX', short_period: str='5', long_period: str='15', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "calculate a simple moving average volume for the ticker and periods input"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/indicator/smav"
    querystring = {}
    if ticker_symbol:
        querystring['ticker_symbol'] = ticker_symbol
    if short_period:
        querystring['short_period'] = short_period
    if long_period:
        querystring['long_period'] = long_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wilder_s_smoothing(ticker_symbol: str='NFLX', period: str='5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "calculate wilder's smoothing for the ticker and periods input"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/indicator/wilder"
    querystring = {}
    if ticker_symbol:
        querystring['ticker_symbol'] = ticker_symbol
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bollinger_bands(period: str='5', ticker_symbol: str='NFLX', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "calculate bollinger bands for the ticker and period input"
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/stock/indicator/bb"
    querystring = {}
    if period:
        querystring['period'] = period
    if ticker_symbol:
        querystring['ticker_symbol'] = ticker_symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ibovespa(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks in the Ibovespa index. See https://en.wikipedia.org/wiki/%C3%8Dndice_Bovespa for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/ibovespa"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bbc_global_30(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks comprising the BBC Global 30 index. See https://en.wikipedia.org/wiki/BBC_Global_30 for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/bbc-global-thirty"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nyse_market_composite(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks comprising the NYSE Market Composite. See http://www.investing.com/indices/nyse-market-composite-components for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/nyse-market-composite"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def amex_oil_index(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The list of stocks comprising the Amex Oil Index. See https://en.wikipedia.org/wiki/Amex_Oil_Index for more details."
    
    """
    url = f"https://stock-market-data.p.rapidapi.com/market/index/amex-oil-index"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_screener(gross_profit_gt: int=104556000000, sector: str='Technology', payout_ratio_lt: int=1, industry: str='Consumer Electronics', exchange: str='NasdaqGS', gross_profit_lt: int=204556000000, country: str='United States', gross_profit_lte: int=204556000000, operating_margin_lte: int=1, dividend_rate_gt: int=0, market_cap_gte: int=2150000000000, payout_ratio_gte: int=0, operating_margin_gte: int=0, price_lt: int=140, payout_ratio_lte: int=1, industry_contains: str='Consumer Electronics', price_book_lte: int=40, float_shares_gte: int=15770636380, price_sales_gte: int=7, debt_equity_lte: int=200, price_earning_gt: int=35, revenue_lt: int=374515000000, price_book_gt: int=33, revenue_growth_lte: int=1, price_gt: int=130, debt_equity_lt: int=200, debt_equity_gte: int=150, current_ratio_lt: int=10, market_cap_lte: int=2350000000000, volume_gt: int=66015000, eps_lt: int=4, price_sales_gt: int=7, earning_growth_gte: int=0, float_shares_lt: int=25770636380, revenue_growth_gt: int=0, payout_ratio_gt: int=0, current_ratio_lte: int=10, profit_margin_lte: int=1, current_ratio_gt: int=1, earning_growth_lt: int=1, dividend_rate_gte: int=0, price_book_gte: int=33, debt_equity_gt: int=150, net_income_gte: int=57410000000, earning_growth_gt: int=0, dividend_rate_lt: int=1, country_contains: str='United States', market_cap_gt: int=2150000000000, price_sales_lte: int=10, float_shares_gt: int=15770636380, operating_margin_lt: int=1, profit_margin_lt: int=1, profit_margin_gt: int=0, net_income_gt: int=57410000000, exchange_contains: str='NasdaqGS', gross_profit_gte: int=104556000000, revenue_gte: int=274515000000, current_ratio_gte: int=1, eps_lte: int=4, revenue_growth_lt: int=1, eps_gt: int=3, net_income_lt: int=67411000000, eps_gte: int=3, profit_margin_gte: int=0, price_book_lt: int=40, price_lte: int=140, revenue_gt: int=27451500000, revenue_growth_gte: int=0, volume_lt: int=76015000, price_earning_lte: int=40, price_earning_gte: int=35, operating_margin_gt: int=0, volume_lte: int=76015000, float_shares_lte: int=25770636380, price_earning_lt: int=40, earning_growth_lte: int=1, volume_gte: int=66015000, market_cap_lt: int=2350000000000, net_income_lte: int=67411000000, sector_contains: str='Technology', dividend_rate_lte: int=1, revenue_lte: int=374515000000, price_gte: int=130, price_sales_lt: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Filtering the full universe of stocks based on custom filters defined on key stock attributes."
    gross_profit_gt: Selects the stocks having gross profit greater than the input value.
        sector: Selects the stocks as per the sector required.
        payout_ratio_lt: Selects the stocks having payout ratio lesser than the input value.
        industry: Selects the stocks as per the industry required.
        exchange: Selects the stocks as per the exchange required.
        gross_profit_lt: Selects the stocks having gross profit lesser than the input value.
        country: Selects the stocks as per the country required.
        gross_profit_lte: Selects the stocks having gross profit lesser than or equal to the input value.
        operating_margin_lte: Selects the stocks having operating margin lesser than or equal to the input value.
        dividend_rate_gt: Selects the stocks having dividend rate greater than the input value.
        market_cap_gte: Selects the stocks having market capital  greater than or equal to the input value.
        payout_ratio_gte: Selects the stocks having payout ratio greater than or equal to the input value.
        operating_margin_gte: Selects the stocks having an operating margin greater than or equal to the input value.
        price_lt: Selects the stocks having price lesser than the input value.
        payout_ratio_lte: Selects the stocks having payout ratio lesser than or equal to the input value.
        industry_contains: Selects the stocks as per the industry required.
        price_book_lte: Selects the stocks having price/book lesser than or equal to the input value.
        float_shares_gte: Selects the stocks having float shares greater than or equal to the input value.
        price_sales_gte: Selects the stocks having price/sales greater than or equal to the input value.
        debt_equity_lte: Selects the stocks having debt/equity lesser than or equal to the input value.
        price_earning_gt: Selects the stocks having price/earning greater than the input value.
        revenue_lt: Selects the stocks having revenue lesser than the required value.
        price_book_gt: Selects the stocks having price/book greater than the input value.
        revenue_growth_lte: Selects the stocks having revenue  growth lesser than or equal to the input value.
        price_gt: Selects the stocks having price greater than the input value.
        debt_equity_lt: Selects the stocks having debt/equity lesser than the input value.
        debt_equity_gte: Selects the stocks having debt/equity greater than or equal to the input value.
        current_ratio_lt: Selects the stocks having current ratio lesser than the input value.
        market_cap_lte: Selects the stocks having market capital  lesser than or equal to the input value.
        volume_gt: Selects the stocks having volume greater than the input value.
        eps_lt: Selects the stocks having eps lesser than the input value.
        price_sales_gt: Selects the stocks having price/sales greater than the input value.
        earning_growth_gte: Selects the stocks having earning growth greater than or equal to the input value.
        float_shares_lt: Selects the stocks having float shares lesser than the input value.
        revenue_growth_gt: Selects the stocks having revenue growth greater than the input value.
        payout_ratio_gt: Selects the stocks having payout ratio greater than the input value.
        current_ratio_lte: Selects the stocks having current ratio lesser than or equal to the input value.
        profit_margin_lte: Selects the stocks having profit margin lesser than or equal to the input value.
        current_ratio_gt: Selects the stocks having current ratio greater than input value.
        earning_growth_lt: Selects the stocks having earning lesser greater than the input value.
        dividend_rate_gte: Selects the stocks having dividend rate greater than or equal to the input value.
        price_book_gte: Selects the stocks having price/book greater than or equal to the input value.
        debt_equity_gt: Selects the stocks having debt/equity greater than the input value.
        net_income_gte: Selects the stocks having net income greater than or equal to the input value.
        earning_growth_gt: Selects the stocks having earning growth greater than the input value.
        dividend_rate_lt: Selects the stocks having dividend rate lesser than the input value.
        country_contains: Selects the stocks as per the country required.
        market_cap_gt: Selects the stocks having market capital  greater than the input value.
        price_sales_lte: Selects the stocks having price/sales lesser than or equal to the input value.
        float_shares_gt: Selects the stocks having float shares greater than the input value.
        operating_margin_lt: Selects the stocks having operating margin lesser than the input value.
        profit_margin_lt: Selects the stocks having profit margin lesser than the input value.
        profit_margin_gt: Selects the stocks having profit margin greater than the input value.
        net_income_gt: Selects the stocks having net income greater than the input value.
        exchange_contains: Selects the stocks as per the exchange required.
        gross_profit_gte: Selects the stocks having gross profit greater than or equal to the input value.
        revenue_gte: Selects the stocks having revenue greater than or equal to the input value.
        current_ratio_gte: Selects the stocks having current ratio greater than or equal to the input value.
        eps_lte: Selects the stocks having eps lesser than or equal to the input value.
        revenue_growth_lt: Selects the stocks having revenue growth lessor than the input value.
        eps_gt: Selects the stocks having eps greater than the input value.
        net_income_lt: Selects the stocks having net income lesser than the input value.
        eps_gte: Selects the stocks having eps greater than or equal to the input value.
        profit_margin_gte: Selects the stocks having profit margin greater than or equal to the input value.
        price_book_lt: Selects the stocks having price/book lesser than the input value.
        price_lte: Selects the stocks having price lesser than or equal to the input value.
        revenue_gt: Selects the stocks having revenue greater than the input value.
        revenue_growth_gte: Selects the stocks having revenue growth greater than or equal to the input value.
        volume_lt: Selects the stocks having volume lesser than the input value.
        price_earning_lte: Selects the stocks having price/earning lesser than or equal to the input value.
        price_earning_gte: Selects the stocks having price/earning greater than or equal to the input value.
        operating_margin_gt: Selects the stocks having operating margin greater than the input value.
        volume_lte: Selects the stocks having volume lesser than or equal to the input value.
        float_shares_lte: Selects the stocks having float shares lesser than or equal to the input value.
        price_earning_lt: Selects the stocks having price/earning lessor than the input value.
        earning_growth_lte: Selects the stocks having earning growth lesser than or equal to the input value.
        volume_gte: Selects the stocks having a volume greater than or equal to the input value.
        market_cap_lt: Selects the stocks having market capital  lesser than the input value.
        net_income_lte: Selects the stocks having net income lesser than or equal to the input value.
        sector_contains: Selects the stocks as per the sector required.
        dividend_rate_lte: Selects the stocks having dividend rate lesser than or equal to the input value.
        revenue_lte: Selects the stocks having revenue lesser than or equal to the input value.
        price_gte: Selects the stocks having a price greater than or equal to the input value.
        price_sales_lt: Selects the stocks having price/sales lesser than the input value.
        
    """
    url = f"https://stock-market-data.p.rapidapi.com/screener"
    querystring = {}
    if gross_profit_gt:
        querystring['gross_profit_gt'] = gross_profit_gt
    if sector:
        querystring['sector'] = sector
    if payout_ratio_lt:
        querystring['payout_ratio_lt'] = payout_ratio_lt
    if industry:
        querystring['industry'] = industry
    if exchange:
        querystring['exchange'] = exchange
    if gross_profit_lt:
        querystring['gross_profit_lt'] = gross_profit_lt
    if country:
        querystring['country'] = country
    if gross_profit_lte:
        querystring['gross_profit_lte'] = gross_profit_lte
    if operating_margin_lte:
        querystring['operating_margin_lte'] = operating_margin_lte
    if dividend_rate_gt:
        querystring['dividend_rate_gt'] = dividend_rate_gt
    if market_cap_gte:
        querystring['market_cap_gte'] = market_cap_gte
    if payout_ratio_gte:
        querystring['payout_ratio_gte'] = payout_ratio_gte
    if operating_margin_gte:
        querystring['operating_margin_gte'] = operating_margin_gte
    if price_lt:
        querystring['price_lt'] = price_lt
    if payout_ratio_lte:
        querystring['payout_ratio_lte'] = payout_ratio_lte
    if industry_contains:
        querystring['industry_contains'] = industry_contains
    if price_book_lte:
        querystring['price_book_lte'] = price_book_lte
    if float_shares_gte:
        querystring['float_shares_gte'] = float_shares_gte
    if price_sales_gte:
        querystring['price_sales_gte'] = price_sales_gte
    if debt_equity_lte:
        querystring['debt_equity_lte'] = debt_equity_lte
    if price_earning_gt:
        querystring['price_earning_gt'] = price_earning_gt
    if revenue_lt:
        querystring['revenue_lt'] = revenue_lt
    if price_book_gt:
        querystring['price_book_gt'] = price_book_gt
    if revenue_growth_lte:
        querystring['revenue_growth_lte'] = revenue_growth_lte
    if price_gt:
        querystring['price_gt'] = price_gt
    if debt_equity_lt:
        querystring['debt_equity_lt'] = debt_equity_lt
    if debt_equity_gte:
        querystring['debt_equity_gte'] = debt_equity_gte
    if current_ratio_lt:
        querystring['current_ratio_lt'] = current_ratio_lt
    if market_cap_lte:
        querystring['market_cap_lte'] = market_cap_lte
    if volume_gt:
        querystring['volume_gt'] = volume_gt
    if eps_lt:
        querystring['eps_lt'] = eps_lt
    if price_sales_gt:
        querystring['price_sales_gt'] = price_sales_gt
    if earning_growth_gte:
        querystring['earning_growth_gte'] = earning_growth_gte
    if float_shares_lt:
        querystring['float_shares_lt'] = float_shares_lt
    if revenue_growth_gt:
        querystring['revenue_growth_gt'] = revenue_growth_gt
    if payout_ratio_gt:
        querystring['payout_ratio_gt'] = payout_ratio_gt
    if current_ratio_lte:
        querystring['current_ratio_lte'] = current_ratio_lte
    if profit_margin_lte:
        querystring['profit_margin_lte'] = profit_margin_lte
    if current_ratio_gt:
        querystring['current_ratio_gt'] = current_ratio_gt
    if earning_growth_lt:
        querystring['earning_growth_lt'] = earning_growth_lt
    if dividend_rate_gte:
        querystring['dividend_rate_gte'] = dividend_rate_gte
    if price_book_gte:
        querystring['price_book_gte'] = price_book_gte
    if debt_equity_gt:
        querystring['debt_equity_gt'] = debt_equity_gt
    if net_income_gte:
        querystring['net_income_gte'] = net_income_gte
    if earning_growth_gt:
        querystring['earning_growth_gt'] = earning_growth_gt
    if dividend_rate_lt:
        querystring['dividend_rate_lt'] = dividend_rate_lt
    if country_contains:
        querystring['country_contains'] = country_contains
    if market_cap_gt:
        querystring['market_cap_gt'] = market_cap_gt
    if price_sales_lte:
        querystring['price_sales_lte'] = price_sales_lte
    if float_shares_gt:
        querystring['float_shares_gt'] = float_shares_gt
    if operating_margin_lt:
        querystring['operating_margin_lt'] = operating_margin_lt
    if profit_margin_lt:
        querystring['profit_margin_lt'] = profit_margin_lt
    if profit_margin_gt:
        querystring['profit_margin_gt'] = profit_margin_gt
    if net_income_gt:
        querystring['net_income_gt'] = net_income_gt
    if exchange_contains:
        querystring['exchange_contains'] = exchange_contains
    if gross_profit_gte:
        querystring['gross_profit_gte'] = gross_profit_gte
    if revenue_gte:
        querystring['revenue_gte'] = revenue_gte
    if current_ratio_gte:
        querystring['current_ratio_gte'] = current_ratio_gte
    if eps_lte:
        querystring['eps_lte'] = eps_lte
    if revenue_growth_lt:
        querystring['revenue_growth_lt'] = revenue_growth_lt
    if eps_gt:
        querystring['eps_gt'] = eps_gt
    if net_income_lt:
        querystring['net_income_lt'] = net_income_lt
    if eps_gte:
        querystring['eps_gte'] = eps_gte
    if profit_margin_gte:
        querystring['profit_margin_gte'] = profit_margin_gte
    if price_book_lt:
        querystring['price_book_lt'] = price_book_lt
    if price_lte:
        querystring['price_lte'] = price_lte
    if revenue_gt:
        querystring['revenue_gt'] = revenue_gt
    if revenue_growth_gte:
        querystring['revenue_growth_gte'] = revenue_growth_gte
    if volume_lt:
        querystring['volume_lt'] = volume_lt
    if price_earning_lte:
        querystring['price_earning_lte'] = price_earning_lte
    if price_earning_gte:
        querystring['price_earning_gte'] = price_earning_gte
    if operating_margin_gt:
        querystring['operating_margin_gt'] = operating_margin_gt
    if volume_lte:
        querystring['volume_lte'] = volume_lte
    if float_shares_lte:
        querystring['float_shares_lte'] = float_shares_lte
    if price_earning_lt:
        querystring['price_earning_lt'] = price_earning_lt
    if earning_growth_lte:
        querystring['earning_growth_lte'] = earning_growth_lte
    if volume_gte:
        querystring['volume_gte'] = volume_gte
    if market_cap_lt:
        querystring['market_cap_lt'] = market_cap_lt
    if net_income_lte:
        querystring['net_income_lte'] = net_income_lte
    if sector_contains:
        querystring['sector_contains'] = sector_contains
    if dividend_rate_lte:
        querystring['dividend_rate_lte'] = dividend_rate_lte
    if revenue_lte:
        querystring['revenue_lte'] = revenue_lte
    if price_gte:
        querystring['price_gte'] = price_gte
    if price_sales_lt:
        querystring['price_sales_lt'] = price_sales_lt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

