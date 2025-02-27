import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def symbols_brief_metrics(category: str, ticker_slugs: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get metrics by Symbol slugs. Separate multiple slugs with a comma"
    ticker_slugs: Ticker Slugs
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/brief-metrics"
    querystring = {'category': category, 'ticker_slugs': ticker_slugs, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_shares(ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get shares for symbol"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/shares"
    querystring = {'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_dividend_history(category: str, ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the dividend history for the symbol"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/dividends/history"
    querystring = {'category': category, 'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_estimated_estimates(estimates_type: str, ticker_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the estimates for the symbol (Latest Quarter's Earnings, Upcoming Quarter's Earnings). Ex result: https://seekingalpha.com/symbol/NVDA/income-statement"
    ticker_id: Ticker ID
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/estimated/estimates"
    querystring = {'estimates_type': estimates_type, 'ticker_id': ticker_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_ratings(ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get symbols ratings for periods. Ratings: authors, quant, sell side. Ex result: https://seekingalpha.com/symbol/NVDA/valuation/metrics"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/rating/periods"
    querystring = {'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_press_releases(ticker_slug: str, page_number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of press releases for symbol. Ex result: https://seekingalpha.com/symbol/NVDA/press-releases"
    ticker_slug: Ticker slug
        page_number: Page number
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/press-releases"
    querystring = {'ticker_slug': ticker_slug, 'page_number': page_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_option_result(month: int, ticker_slug: str, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get option result for the symbol. Ex.: https://seekingalpha.com/symbol/NVDA/options"
    month: Month
        ticker_slug: Ticker slug
        year: Year
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/options/result"
    querystring = {'month': month, 'ticker_slug': ticker_slug, 'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_splits(ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get splits for historical prices. Ex.: https://seekingalpha.com/symbol/NVDA/historical-price-quotes"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/splits"
    querystring = {'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_brief_earnings(ticker_slugs: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get earnings by Symbol slugs. Separate multiple slugs with a comma"
    ticker_slugs: Ticker Slugs
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/brief-earnings"
    querystring = {'ticker_slugs': ticker_slugs, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list(page_number: int, category: str, date_start: str=None, date_end: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of the news by category. Trending news in the /news/trending. Ex.: https://seekingalpha.com/market-news/crypto"
    page_number: Page number
        date_start: Start date, filter data by date range
        date_end: End date, filter data by date range
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/news/list"
    querystring = {'page_number': page_number, 'category': category, }
    if date_start:
        querystring['date_start'] = date_start
    if date_end:
        querystring['date_end'] = date_end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def article_comment_maps(article_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comment maps by article id"
    article_id: Article id
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/articles/comment-maps"
    querystring = {'article_id': article_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_faq(ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Answers to common questions on a topic by Ticker slug. Ex.: https://seekingalpha.com/symbol/NVDA"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/faq"
    querystring = {'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stocks_tickers(screener_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stocks (tickers) from the screener. Ex. https://seekingalpha.com/screeners/96793299-Top-Rated-Stocks"
    screener_id: Screener ID, from `Screeners list` and `Screeners filters`
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/screeners/tickers"
    querystring = {'screener_id': screener_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_attributes(category: str, ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details by Ticker Slugs. Specify attribute category"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/attributes"
    querystring = {'category': category, 'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_sector_metrics(ticker_slug: str, category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the sector metrics for the symbol (Revisions Grade)"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/sector-metrics"
    querystring = {'ticker_slug': ticker_slug, 'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_metric_grades(ticker_slug: str, category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the ticker metric for the symbol"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/metric-grades"
    querystring = {'ticker_slug': ticker_slug, 'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def article_list(category: str, page_number: int, date_end: str=None, date_start: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of the article"
    page_number: Page number
        date_end: End date, filter data by date range
        date_start: Start date, filter data by date range
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/articles/list"
    querystring = {'category': category, 'page_number': page_number, }
    if date_end:
        querystring['date_end'] = date_end
    if date_start:
        querystring['date_start'] = date_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_data(ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get symbol data by Ticker slug, (name, companyName, company, exchange, currency). Ex.: https://seekingalpha.com/symbol/NVDA"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/data"
    querystring = {'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_dividends_payout_ratio(ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get dividends payout ratio chart by Ticker slug"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/dividends/payout-ratio-chart"
    querystring = {'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_ticker_data(ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get symbol ticker data by Ticker slug, (name, companyName, company, exchange, currency). Ex.: https://seekingalpha.com/symbol/NVDA"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/ticker-data"
    querystring = {'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_dividends_estimates(ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get dividends estimates data by Ticker slug"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/dividends/estimates-data"
    querystring = {'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_news(ticker_slug: str, category: str, page_number: int, date_end: str=None, date_start: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of news for symbol. Ex result: https://seekingalpha.com/symbol/NVDA/news"
    ticker_slug: Ticker slug
        page_number: Page number
        date_end: Date end
        date_start: Date start
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/news"
    querystring = {'ticker_slug': ticker_slug, 'category': category, 'page_number': page_number, }
    if date_end:
        querystring['date_end'] = date_end
    if date_start:
        querystring['date_start'] = date_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_analysis(ticker_slug: str, page_number: int, date_start: str=None, date_end: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of analysis for symbol. Ex result: https://seekingalpha.com/symbol/NVDA/analysis"
    ticker_slug: Ticker slug
        page_number: Page number
        date_start: Date start
        date_end: Date end
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/analysis"
    querystring = {'ticker_slug': ticker_slug, 'page_number': page_number, }
    if date_start:
        querystring['date_start'] = date_start
    if date_end:
        querystring['date_end'] = date_end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_sec_filings(page_number: int, category: str, ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of sec filings for symbol. Ex result: https://seekingalpha.com/symbol/NVDA/sec-filings"
    page_number: Page number
        ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/sec-filings"
    querystring = {'page_number': page_number, 'category': category, 'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_ratings_histories(ticker_slug: str, page_number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get symbols histories. Ex result: https://seekingalpha.com/symbol/NVDA/valuation/metrics"
    ticker_slug: Ticker slug
        page_number: Page number
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/rating/histories"
    querystring = {'ticker_slug': ticker_slug, 'page_number': page_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_peers(peers_category: str, ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get peers for the symbol. Ex.: https://seekingalpha.com/symbol/NVDA/peers/related-stocks"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/peers"
    querystring = {'peers_category': peers_category, 'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_related_analysis(page_number: int, ticker_slug: str, date_end: str=None, date_start: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of related analysis for symbol. Ex result: https://seekingalpha.com/symbol/NVDA/related-analysis"
    page_number: Page number
        ticker_slug: Ticker slug
        date_end: Date end
        date_start: Date start
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/related-analysis"
    querystring = {'page_number': page_number, 'ticker_slug': ticker_slug, }
    if date_end:
        querystring['date_end'] = date_end
    if date_start:
        querystring['date_start'] = date_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_metrics(category: str, ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get metrics by Ticker slug. This is the most extensive point, use different groups to get attributes"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/metrics"
    querystring = {'category': category, 'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_historical_prices(date_start: str, show_by: str, date_end: str, ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get historical prices by date and Ticker slug. Ex result: https://seekingalpha.com/symbol/NVDA/valuation/metrics"
    date_start: Date start
        date_end: Date end
        ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/historical-prices"
    querystring = {'date_start': date_start, 'show_by': show_by, 'date_end': date_end, 'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_option_expirations(ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get option expirations for the symbol. Ex.: https://seekingalpha.com/symbol/NVDA/options"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/options/expirations"
    querystring = {'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_author_ratings(ticker_slug: str, page_number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a author ratings for symbol. Ex result: https://seekingalpha.com/symbol/NVDA/ratings/author-ratings"
    ticker_slug: Ticker slug
        page_number: Page number
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/author-ratings"
    querystring = {'ticker_slug': ticker_slug, 'page_number': page_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def meta_tooltips(category: str, ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tooltips by Ticker slug. Getting a description of the data, for each category. Use to get additional data and information."
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/meta/tooltips"
    querystring = {'category': category, 'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_ratings_relative(ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get relative ratings by Ticker slug"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/rating/relative"
    querystring = {'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_advanced(query: str, search_advanced_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all search result (people,symbols,pages) by query. Advanced search"
    query: Query keyword
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/search/advanced"
    querystring = {'query': query, 'search_advanced_type': search_advanced_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_suggested(ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggested tickers by symbol"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/suggested"
    querystring = {'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_transcripts(ticker_slug: str, page_number: int, date_end: str=None, date_start: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of transcripts for symbol. Ex result: https://seekingalpha.com/symbol/NVDA/earnings/transcripts"
    ticker_slug: Ticker slug
        page_number: Page number
        date_end: Date end
        date_start: Date start
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/transcripts"
    querystring = {'ticker_slug': ticker_slug, 'page_number': page_number, }
    if date_end:
        querystring['date_end'] = date_end
    if date_start:
        querystring['date_start'] = date_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feeds_earnings_articles(date_start: str, ticker_slug: str, date_end: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of post earnings articles"
    date_start: Start date, filter data by date range
        ticker_slug: Ticker slug
        date_end: End date, filter data by date range
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/feeds/earnings"
    querystring = {'date_start': date_start, 'ticker_slug': ticker_slug, 'date_end': date_end, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_estimated_earning_announces(ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get estimated earning announces for symbol. Ex result: https://seekingalpha.com/symbol/NVDA/earnings"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/estimated/earning-announces"
    querystring = {'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def article_trending(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of the trending article"
    
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/articles/trending"
    querystring = {'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screeners_details(screener_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get more details about screener. Ex. https://seekingalpha.com/screeners/96793299-Top-Rated-Stocks"
    screener_id: Screener ID, from `Screeners list` and `Screeners filters`
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/screeners/details"
    querystring = {'screener_id': screener_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def press_releases_comment_maps(press_releases_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get press-releases comment maps by Press releases ID. Ex: https://seekingalpha.com/pr/18970018-nvidia-las-vegas-sands-fall-merck-boeing-rise"
    press_releases_id: Press releases ID
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/press-releases/comment-maps"
    querystring = {'press_releases_id': press_releases_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_charts_metrics(chart_period: str, ticker_slug: str, chart_metric: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get charts (metrics) by ticker_slug and period"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/charts-metrics"
    querystring = {'chart_period': chart_period, 'ticker_slug': ticker_slug, 'chart_metric': chart_metric, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def article_details(article_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all details of the article by article id"
    article_id: Article id
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/articles/data"
    querystring = {'article_id': article_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_comments(news_id: int, comment_ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news comments by news_id. Check `News comment-maps`"
    news_id: News ID, from `News trending` or `News list`
        comment_ids: Comment IDs, from `News comment-maps`
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/news/comments"
    querystring = {'news_id': news_id, 'comment_ids': comment_ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def authors_ticker_counts(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ticker counts of author by slug"
    slug: Author slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/authors/ticker-counts"
    querystring = {'slug': slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screeners_filters(screeners_category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of screener filters"
    
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/screeners/filters"
    querystring = {'screeners_category': screeners_category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_data(news_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news data by news_id. Author and news content. Ex.: https://seekingalpha.com/news/3893978-canaan-launches-new-generation-bitcoin-mining-machine"
    news_id: News ID, from `News trending` or `News list`
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/news/data"
    querystring = {'news_id': news_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feeds_earnings_market_current(date_end: str, ticker_slug: str, date_start: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of post earnings market current"
    date_end: End date, filter data by date range
        ticker_slug: Ticker slug
        date_start: Start date, filter data by date range
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/feeds/market-current"
    querystring = {'date_end': date_end, 'ticker_slug': ticker_slug, 'date_start': date_start, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def meta_tooltips_by_slug(slug: str, path: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tooltips for news or article"
    slug: Slug news or article
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/meta/tooltips-by-slug"
    querystring = {'slug': slug, 'path': path, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_suggestion(news_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news suggestion"
    news_id: News ID, from `News trending` or `News list`
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/news/suggestion"
    querystring = {'news_id': news_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_fundamentals(ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fundamentals data by ticker_slug"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/fundamentals"
    querystring = {'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_earning_summaries(ticker_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a earning summaries by symbol"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/earning-summaries"
    querystring = {'ticker_slug': ticker_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_financials_metrics(statement_type: str, ticker_slug: str, currency: str, period_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get financials (income-statement, balance-sheet, cash-flow-statement) for symbol. Ex result: https://seekingalpha.com/symbol/NVDA/income-statement"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/financials/fundamentals-metrics"
    querystring = {'statement_type': statement_type, 'ticker_slug': ticker_slug, 'currency': currency, 'period_type': period_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_trending(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of trending news by category"
    
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/news/trending"
    querystring = {'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def markets_dividend_tr_stocks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of trending dividend stocks"
    
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/markets/dividend-investing"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def article_comments(article_id: int, comment_ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of comments by article id"
    article_id: Article id
        comment_ids: Comment IDs, from `News comment-maps`
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/articles/comments"
    querystring = {'article_id': article_id, 'comment_ids': comment_ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def markets_open(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Getting metadata on open markets"
    
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/markets/open"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_instablog_posts(user_id: int, page_number: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get insta-blog posts of user by user_id"
    user_id: User ID
        page_number: Page number
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/users/instablog/posts"
    querystring = {'user_id': user_id, 'page_number': page_number, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_charts_period(ticker_slug: str, chart_period: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get prices charts by ticker_slug and period"
    ticker_slug: Ticker slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/charts-period"
    querystring = {'ticker_slug': ticker_slug, 'chart_period': chart_period, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_comment_maps(news_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comment-maps for news by news_id"
    news_id: News ID, from `News trending` or `News list`
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/news/comment-maps"
    querystring = {'news_id': news_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feeds_author_articles(page_number: int, author_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of post earnings articles"
    page_number: Page number
        author_slug: Author slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/feeds/author-articles"
    querystring = {'page_number': page_number, 'author_slug': author_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_query(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get search result (people,symbols,pages) by query. Live keyword search by query, short search"
    query: Query keyword
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/search/searches"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_discussion_comments(user_id: int, discussion_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get comments of discussion by user_id and discussion_id"
    user_id: User ID
        discussion_id: Discussion ID
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/users/discussions-comments"
    querystring = {'user_id': user_id, 'discussion_id': discussion_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def markets_global_indices(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get global indices"
    
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/markets/global-indices"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def instablogs_post_data(instablog_post_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instablogs post data by instablog_post_id"
    instablog_post_id: Instablog post ID
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/instablogs/data"
    querystring = {'instablog_post_id': instablog_post_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screeners_list(screeners_category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of screeners, (Top Rated Stocks, Top Quant Dividend Stocks, Most Shorted Stocks). Ex. https://seekingalpha.com/screeners"
    
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/screeners/list"
    querystring = {'screeners_category': screeners_category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def markets_equities_groups(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get equities groups, (global-equity, countries-equity)"
    
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/markets/equities-groups"
    querystring = {'category': category, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def markets_day_watch(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get day-watch, top gainers, top losers, cryptocurrencies, most active"
    
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/markets/day-watch"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def authors_details(author_slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all details of author by the author slug"
    author_slug: Author slug
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/authors/data"
    querystring = {'author_slug': author_slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def press_releases_data(press_releases_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get press releases data by Press releases ID. Use `Symbols Press releases` endpoint. Ex: https://seekingalpha.com/pr/18970018-nvidia-las-vegas-sands-fall-merck-boeing-rise"
    press_releases_id: Press releases ID
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/press-releases/data"
    querystring = {'press_releases_id': press_releases_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def instablogs_comment_maps(instablog_post_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get instablogs comments map by instablog_post_id"
    instablog_post_id: Instablog post ID
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/instablogs/comment-maps"
    querystring = {'instablog_post_id': instablog_post_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filings_comment_maps(filing_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get filing comment maps by Filing ID. Use `Symbols Sec Filings` endpoint. Ex: https://seekingalpha.com/filing/6823254"
    filing_id: Filing ID
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/filings/comment-maps"
    querystring = {'filing_id': filing_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_quotes(symbol_ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get real time quotes by Symbol IDs. For ex: Nasdaq,DowJones (590019,598187). BTC-USD (581328). Separate multiple IDs with a comma. ![Screen](https://i.imgur.com/RPLPfUU.png)"
    symbol_ids: Symbol IDs
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/symbols/quotes"
    querystring = {'symbol_ids': symbol_ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filings_data(filing_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get filing data by Filing ID. Use `Symbols Sec Filings` endpoint. Ex: https://seekingalpha.com/filing/6823254"
    filing_id: Filing ID
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/filings/data"
    querystring = {'filing_id': filing_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_discussions(page_number: int, user_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get discussions posts of user by user_id"
    page_number: Page number
        user_id: User ID
        
    """
    url = f"https://seeking-alpha-finance.p.rapidapi.com/v1/users/discussions"
    querystring = {'page_number': page_number, 'user_id': user_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

