import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stock_quote(symbol: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock market quote. Supports all stock types: *stock*, *index*, *mutual fund* and *futures*."
    symbol: Stock symbol / ticker.

**e.g.** *`MSFT:NASDAQ`*
**e.g.** *`MSFT`*
**e.g.** *`^DJI`*
**e.g.** *`VTSAX`*
        language: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

**Default**: *en*.
        
    """
    url = f"https://real-time-finance-data.p.rapidapi.com/stock-quote"
    querystring = {'symbol': symbol, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-finance-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_news(from_symbol: str, to_symbol: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest news related to a specific currency / forex or crypto."
    from_symbol: A 3-Letter currency code / symbol (ISO 4217). For example: *USD*.
        to_symbol: A 3-Letter currency code / symbol (ISO 4217). For example: *EUR*.
        language: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

**Default**: *en*.
        
    """
    url = f"https://real-time-finance-data.p.rapidapi.com/currency-news"
    querystring = {'from_symbol': from_symbol, 'to_symbol': to_symbol, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-finance-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_exchange_rate(from_symbol: str, to_symbol: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get currency / forex or crypto exchange rates."
    from_symbol: A 3-Letter currency code / symbol (ISO 4217) to convert. For example: *USD*.
        to_symbol: A 3-Letter currency code / symbol (ISO 4217) to convert to. For example: *EUR*.
        language: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

**Default**: *en*.
        
    """
    url = f"https://real-time-finance-data.p.rapidapi.com/currency-exchange-rate"
    querystring = {'from_symbol': from_symbol, 'to_symbol': to_symbol, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-finance-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_time_series(period: str, to_symbol: str, from_symbol: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get forex time series for the following periods: **1 day**, **5 days**, **1 month**, **6 months**, **year-to-date**, **1 year**, **5 years** and **all times (max)**."
    period: Period for which to return time series.

**Periods:**
- **1D** - 1 day.
- **5D** - 5 days.
- **1M** - 1 month.
- **6M** - 6 months.
- **YTD** - year-to-date.
- **1Y** - 1 year.
- **5Y** - 5 years.
- **MAX** - all times.
        to_symbol: A 3-Letter currency code / symbol (ISO 4217). For example: *EUR*.
        from_symbol: A 3-Letter currency code / symbol (ISO 4217). For example: *USD*.
        language: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

**Default**: *en*.
        
    """
    url = f"https://real-time-finance-data.p.rapidapi.com/currency-time-series"
    querystring = {'period': period, 'to_symbol': to_symbol, 'from_symbol': from_symbol, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-finance-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_company_overview(symbol: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock / public company details. Supports all stock types: *stock*, *index*, *mutual fund* and *futures*. Returns company details for the *stock* type."
    symbol: Stock symbol (ticker).

**e.g.** *`MSFT:NASDAQ`*
**e.g.** *`MSFT`*
**e.g.** *`^DJI`*
**e.g.** *`VTSAX`*
        language: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

**Default**: *en*.
        
    """
    url = f"https://real-time-finance-data.p.rapidapi.com/stock-overview"
    querystring = {'symbol': symbol, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-finance-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_cash_flow(period: str, symbol: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get public company's **quarterly** or **annual** cash flow information. Supports the *stock* type only."
    period: Period for which to get company's cash flow.

**Periods**: *QUARTERLY*, *ANNUAL*.
        symbol: Stock symbol (ticker).

**e.g.** *`MSFT:NASDAQ`*
**e.g.** *`MSFT`*
**e.g.** *`AAPL`*
        language: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

**Default**: *en*.
        
    """
    url = f"https://real-time-finance-data.p.rapidapi.com/company-cash-flow"
    querystring = {'period': period, 'symbol': symbol, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-finance-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_balance_sheet(period: str, symbol: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get public company's **quarterly** or **annual** balance sheet. Supports the *stock* type only."
    period: Period for which to get company's balance sheet.

**Periods**: *QUARTERLY*, *ANNUAL*.
        symbol: Stock symbol (ticker).

**e.g.** *`MSFT:NASDAQ`*
**e.g.** *`MSFT`*
**e.g.** *`AAPL`*
        language: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

**Default**: *en*.
        
    """
    url = f"https://real-time-finance-data.p.rapidapi.com/company-balance-sheet"
    querystring = {'period': period, 'symbol': symbol, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-finance-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_income_statement(symbol: str, period: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get public company's **quarterly** or **annual** income statement. Supports the *stock* type only."
    symbol: Stock symbol (ticker).

**e.g.** *`MSFT:NASDAQ`*
**e.g.** *`MSFT`*
**e.g.** *`AAPL`*
        period: Period for which to get company's income statement.

**Periods**: *QUARTERLY*, *ANNUAL*.
        language: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

**Default**: *en*.
        
    """
    url = f"https://real-time-finance-data.p.rapidapi.com/company-income-statement"
    querystring = {'symbol': symbol, 'period': period, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-finance-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_news(symbol: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest news related to a specific stock.  Supports all stock types: *stock*, *index*, *mutual fund* and *futures*."
    symbol: Stock symbol / ticker.

**e.g.** *`MSFT:NASDAQ`*
**e.g.** *`MSFT`*
**e.g.** *`^DJI`*
**e.g.** *`VTSAX`*
        language: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

**Default**: *en*.
        
    """
    url = f"https://real-time-finance-data.p.rapidapi.com/stock-news"
    querystring = {'symbol': symbol, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-finance-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_time_series(period: str, symbol: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock time series and key news events for the following periods: **1 day**, **5 days**, **1 month**, **6 months**, **year-to-date**, **1 year**, **5 years** and **all times (max)**. In addition, the endpoints returns the key news events that affected the stock prices  in the specified period. Supports all stock types: *stock*, *index*, *mutual fund* and *futures*."
    period: Period for which to return time series and key events.

**Periods:**
- **1D** - 1 day.
- **5D** - 5 days.
- **1M** - 1 month.
- **6M** - 6 months.
- **YTD** - year-to-date.
- **1Y** - 1 year.
- **5Y** - 5 years.
- **MAX** - all times.
        symbol: Stock symbol (ticker).

**e.g.** *`MSFT:NASDAQ`*
**e.g.** *`MSFT`*
**e.g.** *`^DJI`*
**e.g.** *`VTSAX`*
        language: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

**Default**: *en*.
        
    """
    url = f"https://real-time-finance-data.p.rapidapi.com/stock-time-series"
    querystring = {'period': period, 'symbol': symbol, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-finance-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(query: str, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find stocks, indices, mutual funds, futures, currency / forex / crypto using a free-form query or symbol as seen on Google Finance - https://www.google.com/finance."
    query: Free-form search query.

**e.g.** *`AAPL`*
**e.g.** *`Microsoft NASDAQ`*
**e.g.** *`Dow Johns`*
**e.g.** *`USD to EUR`*
        language: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

**Default**: *en*.
        
    """
    url = f"https://real-time-finance-data.p.rapidapi.com/search"
    querystring = {'query': query, }
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-finance-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_trends(trend_type: str, country: str='us', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the latest market trends and relevant news. Supported trends: **Most Active**, **Gainers**, **Losers**, **Crypto**, **Currencies** and **Climate Leaders**."
    trend_type: Trend type.

**Supported trend types:**

- *MARKET_INDEXES*
- *MOST_ACTIVE*
- *GAINERS*
- *LOSERS*
- *CRYPTO*
- *CURRENCIES*
- *CLIMATE_LEADERS*
        country: The country for which to get trends, specified as a 2-letter country code - see [ISO 3166](https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes).

**Default**: *us*.
        language: The language to use for the results, specified as a 2-letter language code - see [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes).

**Default**: *en*.
        
    """
    url = f"https://real-time-finance-data.p.rapidapi.com/market-trends"
    querystring = {'trend_type': trend_type, }
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-time-finance-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

