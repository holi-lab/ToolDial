import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stock_get_earnings_per_share(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return earnings per share"
    symbol: Ticker Symbol ( Ex. \\"AAPL\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_earnings_per_share"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_net_income(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return net income"
    symbol: Ticker Symbol ( Ex. \\"AAPL\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_net_income"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_ebit(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return ebit"
    symbol: Ticker Symbol ( Ex. \\"AAPL\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_ebit"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_price_to_sales(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return price to sales"
    symbol: Ticker Symbol ( Ex. \\"AAPL\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_price_to_sales"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_pe_ratio(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return pe ratio"
    symbol: Ticker Symbol ( Ex. \\"AAPL\\" is Ticker symbol for Apple Inc. on the stock market )
        region: US
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_pe_ratio"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_payout_ratio(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return payout ratio"
    symbol: Ticker Symbol ( Ex. \\"AAPL\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_payout_ratio"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_beta(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return beta"
    symbol: Ticker Symbol ( Ex. \\"AAPL\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_beta"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_200day_moving_avg(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return 200 day moving average"
    symbol: Ticker Symbol ( Ex. \\"AAPL\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_200day_moving_avg"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_50day_moving_avg(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return 50 day moving average"
    symbol: Ticker Symbol ( Ex. \\"AAPL\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_50day_moving_avg"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_annual_avg_div_rate(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return annual average dividend rate"
    symbol: Ticker Symbol ( Ex. \\"AAPL\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_annual_avg_div_rate"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_dividend_rate(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return dividend rate"
    symbol: Ticker Symbol ( Ex. \\"AAPL\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_dividend_rate"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_five_yr_avg_div_yield(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return five year average dividend yield"
    symbol: Ticker Symbol ( Ex. \\\\\\\\\"AAPL\\\\\\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_five_yr_avg_div_yield"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_annual_avg_div_yield(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return annual avg div yield"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_annual_avg_div_yield"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_dividend_yield(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return dividend yield"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_dividend_yield"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_yearly_low(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return yearly low"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_yearly_low"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_yearly_high(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return yearly high"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_yearly_high"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_currency(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return get currency"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: US
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_currency"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_daily_high(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return daily h igh"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: US
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_daily_high"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_daily_low(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return daily low"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_daily_low"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_market_cap(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return market cap"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_market_cap"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_stock_exchange(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return stock exchange"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_stock_exchange"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_ten_day_avg_daily_volume(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return ten day avg daily volume"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_ten_day_avg_daily_volume"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_open_price(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return open price"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_open_price"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_prev_close_price(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return previous close price"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_prev_close_price"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_current_volume(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return current volume"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_current_volume"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_current_percent_change(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return current percent change"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_current_percent_change"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_current_change(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return current change"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_current_change"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_gross_profit(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return gross profit"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_gross_profit"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_cost_of_revenue(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return cost of revenue"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: US
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_cost_of_revenue"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_total_revenue(symbol: str, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return total revenue"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_total_revenue"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_operating_income(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return operatin income"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_operating_income"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_interest_expense(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return interest expense for stock"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_interest_expense"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_research_and_development(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return research and development"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_research_and_development"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_daily_dividend_data(symbol: str, end_date: str, start_date: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return daily dividend data"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        end_date: end_date should be entered in the 'YYYY-MM-DD' format and is the last day that data will be pulled for.

        start_date: start_date should be entered in the 'YYYY-MM-DD' format and is the first day that data will be pulled for.
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_daily_dividend_data"
    querystring = {'symbol': symbol, 'end_date': end_date, 'start_date': start_date, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_key_statistics_data(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return key statistics data"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_key_statistics_data"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_financial_data(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return stock financial information"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_financial_data"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_stock_profile_data(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return stock profile data"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_stock_profile_data"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_num_shares_outstanding(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return number of share outstanding"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_num_shares_outstanding"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_stock_historical_price_data(symbol: str, start_date: str, time_interval: str, end_date: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method will pull historical pricing data for stocks, currencies, ETFs, mutual funds, U.S. Treasuries, cryptocurrencies, commodities, and indexes."
    symbol: Ticker Symbol ( Ex. \\\\\\\\\"AAPL\\\\\\\\\" is Ticker symbol for Apple Inc. on the stock market )
        start_date: start_date should be entered in the 'YYYY-MM-DD' format and is the first day that data will be pulled for.

        time_interval: time_interval can be either 'daily', 'weekly', or 'monthly'
        end_date: end_date should be entered in the 'YYYY-MM-DD' format and is the last day that data will be pulled for.

        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_stock_historical_price_data"
    querystring = {'symbol': symbol, 'start_date': start_date, 'time_interval': time_interval, 'end_date': end_date, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_stock_quote_type_data(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return stock quota type data"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_stock_quote_type_data"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_stock_earnings_data(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return stock earning data"
    symbol: Ticker Symbol ( Ex. \\\\\\\\\"AAPL\\\\\\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_stock_earnings_data"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_financial_stmts(frequency: str, symbol: str, statement_type: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return financial statements for 'income', 'balance',  or 'cash'"
    frequency: frequency can be either 'annual' or 'quarterly'.

        symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        statement_type: statement_type can be 'income', 'balance', 'cash' or a list of several.

        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_financial_stmts"
    querystring = {'frequency': frequency, 'symbol': symbol, 'statement_type': statement_type, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_stock_price(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "return current stock price"
    symbol: Ticker Symbol ( Ex. \\\\"AAPL\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_stock_price"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_summary(symbol: str, region: str='US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns financial summary data for cryptocurrencies, stocks, currencies, ETFs, mutual funds, U.S. Treasuries, commodity futures, and indexes."
    symbol: Ticker Symbol   ( Ex. \\\\\\\"AAPL\\\\\\\" is Ticker symbol for Apple Inc. on the stock market )
        region: One of the following is allowed 
US|BR|AU|CA|FR|DE|HK|IN|IT|ES|GB|S
        
    """
    url = f"https://yh-finance8.p.rapidapi.com/stock/get_summary"
    querystring = {'symbol': symbol, }
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yh-finance8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

