import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_v1_news_articles_bounded_upper_bound(upper_bound: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of all News Upper Bound is an Integer indicating a total number of articles to return"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/news/articles-bounded/{upper_bound}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_eod_date_exchange_code_stock_code(date: str, exchange_code: str, stock_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "will return end of day data for a certain stock at a given date"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/eod/{date}/{exchange_code}.{stock_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_sentiment_trending_stock_stock_code(stock_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return a sentiment for most trending Tweets by stock"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/sentiment/trending/stock/{stock_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_eod_from_to_stock_code(stock_code: str, is_from: str, to: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "will return end of day data for a certain stock at a given date"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/eod/{from}.{to}/{stock_code}"
    querystring = {'_from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_news_articles_by_ticker_stock_code(stock_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Financial News Articles By Ticker"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/news/articles-by-ticker/{stock_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_sentiment_tweet_stock_stock_code(stock_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return a list Social Media Sentiments for a certain stock"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/sentiment/tweet/stock/{stock_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_sentiment_trend_setters_stock_stock_code(stock_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given the stock_code return a list of trend setters for this stock and their sentiments"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/sentiment/trend-setters/stock/{stock_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_news_articles_by_date_date(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Articles Financial News By Date"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/news/articles-by-date/{date}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_stocks_options_stock_stock_code(stock_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get Stock options data for the provided stock code"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/stocks/options/stock/{stock_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_stocks_contract_call_put_id(call_put_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "given call_put_id return Contract"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/stocks/contract/{call_put_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_exchange_outstanding_shares_exchange_code_exchange_code_year(year: str, exchange_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given an exchange_code and a year return all outstanding shares for all companies for that year"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/exchange-outstanding-shares/exchange-code/{exchange_code}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_quarterly_balance_sheet_filing_date_stock_code(filing_date: str, stock_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "given filing_date and balance_sheet_id return Quarterly Balance Sheet"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/quarterly-balance-sheet/{filing_date}/{stock_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_financial_statements_company_statement_stock_code_year(stock_code: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "given an a stock_code and a year return a complete financial statements for the year"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/financial-statements/company-statement/{stock_code}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_news_article_uuid(uuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Obtain Financial News Information related to listed companies"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/news/article/{uuid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_highlights_stock_stock_code(stock_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get fundamental highlights data from either fundamental_id or stock_codes"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/highlights/stock/{stock_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_financial_statements_exchange_year_exchange_code_year(year: str, exchange_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given an exchange_code and a year return a complete list of financial statements for all companies in exchange"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/financial-statements/exchange-year/{exchange_code}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_exchange_listed_companies_exchange_code(exchange_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns a list of listed companies in exchange_code"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/exchange/listed-companies/{exchange_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_exchange_id_exchange_id(exchange_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "given exchange_id or exchange_code code will return exchange_code data"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/exchange/id/{exchange_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_tech_indicators_by_company_stock_code_stock_code_year(stock_code: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a company stock_code and a year return technical indicators for that year"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/tech-indicators-by-company/stock-code/{stock_code}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_exchange_exchange_with_tickers_code_exchange_code(exchange_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Exchange Data with a total list of stock_codes and stock_id's "
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/exchange/exchange-with-tickers/code/{exchange_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_exchange_code_exchange_code(exchange_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "given exchange_id or exchange_code code will return exchange_code data"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/exchange/code/{exchange_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_financials_income_statements_statement_id(statement_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Will return Company Income Statement By Statement ID"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/financials/income-statements/{statement_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_company_insider_transactions_stock_code_stock_code_year(year: str, stock_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given companies stock_code, and year return a list of insider transactions"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/company-insider-transactions/stock-code/{stock_code}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_company_address_id_fundamental_id(fundamental_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns company address data"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/company-address/id/{fundamental_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_company_details_stock_stock_code(stock_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns company details from company fundamental data"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/company-details/stock/{stock_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamental_general(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return general fundamental data for all companies"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamental/general"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_company_details_id_fundamental_id(fundamental_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns company details from company fundamental data"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/company-details/id/{fundamental_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_financial_statements_filing_date_ticker_filing_date_stock_code(stock_code: str, filing_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Income Statements by Filing Date and Company Stock Code"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/financial-statements/filing-date-ticker/{filing_date}/{stock_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_annual_balance_sheet_filing_date_stock_code(filing_date: str, stock_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "given the filing date & balance_sheet_id return annual balance sheet"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/annual-balance-sheet/{filing_date}/{stock_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_stocks_country_country(country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns a list of stocks listed in a certain country"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/stocks/country/{country}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_exchanges(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns a list of exchanges"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/exchanges"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_stocks_currency_currency(currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns a list of stocks listed with a certain currency"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/stocks/currency/{currency}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_stocks_exchange_code_exchange_code(exchange_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "given an exchange_code code return a list of stocks in the exchange_code"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/stocks/exchange/code/{exchange_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamental_company_stock_code(stock_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "for each stock code return the fundamental data for the company"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamental/company/{stock_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_stock_code_stock_code(stock_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "given stock code return stock details"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/stock/code/{stock_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_financial_statements_ticker_date_range_from_to_stock_code(stock_code: str, to: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given two dates and a stock date return all financial statements, _from & _to should be date-strings"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/financial-statements/ticker-date-range/{from}.{to}/{stock_code}"
    querystring = {'_from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_financial_statements_by_term_from_to_stock_code_term(is_from: str, stock_code: str, to: str, term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return Annual or Quarterly Statements based on Term, Stock Code, and dates _from and _to"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/financial-statements/by-term/{from}.{to}/{stock_code}/{term}"
    querystring = {'_from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_tech_indicators_by_exchange_exchange_code_exchange_code_year(year: str, exchange_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given an exchange_code and a year return all technical indicators released for companies listed on the exchange for the year mentioned"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/tech-indicators-by-exchange/exchange-code/{exchange_code}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_exchange_listed_stocks_exchange_code(exchange_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns a list of listed stocks in exchange_code"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/exchange/listed-stocks/{exchange_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_company_address_stock_stock_code(stock_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns company address data"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/company-address/stock/{stock_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_company_valuations_stock_code_stock_code_year(stock_code: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Company Valuations Data for the year"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/company-valuations/stock-code/{stock_code}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_exchange_analyst_rankings_exchange_code_exchange_code_year(year: str, exchange_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Analyst rankings for all companies listed under a specific exchange"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/exchange-analyst-rankings/exchange-code/{exchange_code}/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_fundamentals_highlights_id_fundamental_id(fundamental_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get fundamental highlights data from either fundamental_id or stock_codes"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/fundamentals/highlights/id/{fundamental_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_stocks_exchange_id_exchange_id(exchange_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "given an exchange_code id return a list of stocks in the exchange_code"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/stocks/exchange/id/{exchange_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_stocks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns a complete list of stocks present"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/stocks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_eod_from_to_exchange_code(is_from: str, exchange_code: str, to: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return a list of eod historical data on exchange_code from one date to the other"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/eod/{from}.{to}/{exchange_code}"
    querystring = {'_from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_eod_date_exchange_code(date: str, exchange_code: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "will return end of day data for a certain stock at a given date"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/eod/{date}/{exchange_code}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def api_v1_news_articles_by_publisher_publisher(publisher: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get News Financial News by publisher"
    
    """
    url = f"https://stock-api6.p.rapidapi.com/api/v1/news/articles-by-publisher/{publisher}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-api6.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

