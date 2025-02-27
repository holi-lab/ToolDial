import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ratings(symbol: str, country: str=None, dp: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API request returns ratings of a mutual fund.
		
		Availability: Mega plan
		Data weighting: 200 API credits per symbol"
    symbol: Symbol ticker of mutual fund
        country: Filter by country name or alpha code
        dp: Number of decimal places for floating values
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/mutual_funds/world/ratings"
    querystring = {'symbol': symbol, }
    if country:
        querystring['country'] = country
    if dp:
        querystring['dp'] = dp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def composition(symbol: str, country: str=None, dp: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API request returns the portfolio composition of a mutual fund, including sectors, holding details, weighted exposure, and others.
		
		Availability: Mega plan
		Data weighting: 200 API credits per symbol"
    symbol: Symbol ticker of mutual fund
        country: Filter by country name or alpha code
        dp: Number of decimal places for floating values
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/mutual_funds/world/composition"
    querystring = {'symbol': symbol, }
    if country:
        querystring['country'] = country
    if dp:
        querystring['dp'] = dp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def purchase_info(symbol: str, dp: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API request returns essential information on purchasing a mutual fund, including minimums, pricing, and available brokerages.
		
		Availability: Mega plan
		Data weighting: 200 API credits per symbol"
    symbol: Symbol ticker of mutual fund
        dp: Number of decimal places for floating values
        country: Filter by country name or alpha code
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/mutual_funds/world/purchase_info"
    querystring = {'symbol': symbol, }
    if dp:
        querystring['dp'] = dp
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eps_revisions(symbol: str, exchange: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns analysts’ revisions of a company's future quarterly and annual earnings per share (EPS) over the last week and month.
		
		Availability: Mega plan
		Data weighting: 20 API credits per symbol"
    exchange: Filter by exchange name or mic code
        country: Filter by country name or alpha code
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/eps_revisions"
    querystring = {'symbol': symbol, }
    if exchange:
        querystring['exchange'] = exchange
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def growth_estimates(symbol: str, exchange: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns consensus analyst estimates over the company's growth rates for various periods.
		
		Availability: Mega plan
		Data weighting: 20 API credits per symbol"
    exchange: Filter by exchange name or mic code
        country: Filter by country name or alpha code
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/growth_estimates"
    querystring = {'symbol': symbol, }
    if exchange:
        querystring['exchange'] = exchange
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def analyst_ratings_us_equities(symbol: str, exchange: str=None, country: str=None, rating_change: str=None, outputsize: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns complete information on ratings issued by analyst firms. Works only for US equities.
		
		Availability: Mega plan
		Data weighting: 200 API credits per symbol"
    exchange: Filter by exchange name or mic code
        country: Filter by country name or alpha code
        rating_change: Filter by rating change action: `Maintains`, `Upgrade`, `Downgrade`, `Initiates` or `Reiterates`
        outputsize: Number of records in response; default `30`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/analyst_ratings/us_equities"
    querystring = {'symbol': symbol, }
    if exchange:
        querystring['exchange'] = exchange
    if country:
        querystring['country'] = country
    if rating_change:
        querystring['rating_change'] = rating_change
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def analyst_ratings_light(symbol: str, exchange: str=None, rating_change: str=None, outputsize: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns a lightweight version of ratings issued by analyst firms. Works for US and international markets.
		
		Availability: Mega plan
		Data weighting: 75 API credits per symbol"
    exchange: Filter by exchange name or mic code
        rating_change: Filter by rating change action: `Maintains`, `Upgrade`, `Downgrade`, `Initiates` or `Reiterates`
        outputsize: Number of records in response; default `30`
        country: Filter by country name or alpha code
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/analyst_ratings/light"
    querystring = {'symbol': symbol, }
    if exchange:
        querystring['exchange'] = exchange
    if rating_change:
        querystring['rating_change'] = rating_change
    if outputsize:
        querystring['outputsize'] = outputsize
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def price_target(symbol: str, country: str=None, exchange: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns the analysts' projection of a security's future price.
		
		Availability: Mega plan
		Data weighting: 75 API credits per symbol"
    country: Filter by country name or alpha code
        exchange: Filter by exchange name or mic code
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/price_target"
    querystring = {'symbol': symbol, }
    if country:
        querystring['country'] = country
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def recommendations(symbol: str, exchange: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns the average of all analyst recommendations and classifies them as Strong Buy, Buy, Hold, or Sell. Also, it returns a recommendation score.
		
		Availability: Mega plan
		Data weighting: 100 API credits per symbol"
    exchange: Filter by exchange name or mic code
        country: Filter by country name or alpha code
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/recommendations"
    querystring = {'symbol': symbol, }
    if exchange:
        querystring['exchange'] = exchange
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def eps_trend(symbol: str, exchange: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns a breakdown of the estimated historical EPS changes at a given period.
		
		Availability: Mega plan
		Data weighting: 20 API credits per symbol"
    exchange: Filter by exchange name or mic code
        country: Filter by country name or alpha code
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/eps_trend"
    querystring = {'symbol': symbol, }
    if exchange:
        querystring['exchange'] = exchange
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def revenue_estimate(symbol: str, exchange: str=None, country: str=None, dp: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns analysts' estimate for a company's future quarterly and annual sales (total revenue).
		
		Availability: Mega plan
		Data weighting: 20 API credits per symbol"
    exchange: Filter by exchange name or mic code
        country: Filter by country name or alpha code
        dp: • Number of decimal places for floating values 
• Should be in range `[0,11]` inclusive. Default `5`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/revenue_estimate"
    querystring = {'symbol': symbol, }
    if exchange:
        querystring['exchange'] = exchange
    if country:
        querystring['country'] = country
    if dp:
        querystring['dp'] = dp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earnings_estimate(symbol: str, country: str=None, exchange: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API endpoint returns analysts' estimate for a company's future quarterly and annual earnings per share (EPS).
		
		Availability: Mega plan
		Data weighting: 20 API credits per symbol"
    country: Filter by country name or alpha code
        exchange: Filter by exchange name or mic code
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/earnings_estimate"
    querystring = {'symbol': symbol, }
    if country:
        querystring['country'] = country
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def institutional_holders(symbol: str, country: str=None, exchange: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the amount of the company’s available stock owned by institutions (pension funds, insurance companies, investment firms, private foundations, endowments, or other large entities that manage funds on behalf of others).
		
		Availability: Mega plan
		Data weighting: 1500 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/institutional_holders"
    querystring = {'symbol': symbol, }
    if country:
        querystring['country'] = country
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fund_holders(symbol: str, country: str=None, exchange: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the amount of the company’s available stock owned by mutual fund holders.
		
		Availability: Mega plan
		Data weighting: 1500 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/fund_holders"
    querystring = {'symbol': symbol, }
    if country:
        querystring['country'] = country
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def options_expiration(symbol: str, exchange: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the expiration dates of an option contract.
		
		Availability: Ultra plan
		Data weighting: 200 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/options/expiration"
    querystring = {'symbol': symbol, }
    if exchange:
        querystring['exchange'] = exchange
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cash_flow(symbol: str, exchange: str=None, country: str=None, period: str='annual', end_date: str=None, start_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns complete cash flow of a company showing net the amount of cash and cash equivalents being transferred into and out of a business.
		
		Availability: Ultra plan
		Data weighting: 100 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/cash_flow"
    querystring = {'symbol': symbol, }
    if exchange:
        querystring['exchange'] = exchange
    if country:
        querystring['country'] = country
    if period:
        querystring['period'] = period
    if end_date:
        querystring['end_date'] = end_date
    if start_date:
        querystring['start_date'] = start_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def balance_sheet(symbol: str, exchange: str=None, country: str=None, start_date: str=None, end_date: str=None, period: str='annual', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns complete balance sheet of a company showing the summary of assets, liabilities, and shareholders’ equity.
		
		Availability: Ultra plan
		Data weighting: 100 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/balance_sheet"
    querystring = {'symbol': symbol, }
    if exchange:
        querystring['exchange'] = exchange
    if country:
        querystring['country'] = country
    if start_date:
        querystring['start_date'] = start_date
    if end_date:
        querystring['end_date'] = end_date
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def income_statement(symbol: str, period: str='annual', exchange: str=None, start_date: str=None, country: str=None, end_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns complete income statement of a company and shows the company’s revenues and expenses during a period (annual or quarter).
		
		Availability: Ultra plan
		Data weighting: 100 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/income_statement"
    querystring = {'symbol': symbol, }
    if period:
        querystring['period'] = period
    if exchange:
        querystring['exchange'] = exchange
    if start_date:
        querystring['start_date'] = start_date
    if country:
        querystring['country'] = country
    if end_date:
        querystring['end_date'] = end_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insider_transactions(symbol: str=None, exchange: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns trading information performed by insiders.
		
		Availability: Ultra plan
		Data weighting: 200 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/insider_transactions"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    if exchange:
        querystring['exchange'] = exchange
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earnings_calendar(format: str=None, end_date: str=None, start_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API method returns earning data as a calendar for a given date range. By default today’s earning is returned. To call a custom period, use start_date and end_date parameters.
		
		Availability: Pro plan
		Data weighting: 40 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/earnings_calendar"
    querystring = {}
    if format:
        querystring['format'] = format
    if end_date:
        querystring['end_date'] = end_date
    if start_date:
        querystring['start_date'] = start_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earnings(symbol: str, format: str=None, outputsize: str=None, type: str=None, exchange: str=None, country: str=None, period: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API call returns earnings data for a given company, including EPS estimate and EPS actual. Earnings are available for complete company history.
		
		Availability: Pro plan
		Data weighting: 20 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/earnings"
    querystring = {'symbol': symbol, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if type:
        querystring['type'] = type
    if exchange:
        querystring['exchange'] = exchange
    if country:
        querystring['country'] = country
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def splits(symbol: str, start_date: str=None, country: str=None, exchange: str=None, range: str='6m', end_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the date and the split factor of shares of the company for the last 10+ years.
		
		Availability: Pro plan
		Data weighting: 20 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/splits"
    querystring = {'symbol': symbol, }
    if start_date:
        querystring['start_date'] = start_date
    if country:
        querystring['country'] = country
    if exchange:
        querystring['exchange'] = exchange
    if range:
        querystring['range'] = range
    if end_date:
        querystring['end_date'] = end_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dividends(symbol: str, start_date: str=None, end_date: str=None, country: str=None, range: str='6m', exchange: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the amount of dividends paid out for the last 10+ years.
		
		Availability: Pro plan
		Data weighting: 20 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/dividends"
    querystring = {'symbol': symbol, }
    if start_date:
        querystring['start_date'] = start_date
    if end_date:
        querystring['end_date'] = end_date
    if country:
        querystring['country'] = country
    if range:
        querystring['range'] = range
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile(symbol: str, exchange: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns general information about the company.
		
		Availability: Pro plan
		Data weighting: 10 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/profile"
    querystring = {'symbol': symbol, }
    if exchange:
        querystring['exchange'] = exchange
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def logo(symbol: str, country: str=None, exchange: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns logo of the company.
		
		Availability: Basic plan
		Data weighting: 1 API credit per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/logo"
    querystring = {'symbol': symbol, }
    if country:
        querystring['country'] = country
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sustainability(symbol: str, dp: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API request returns brief information on mutual fund sustainability and ESG.
		
		Availability: Mega plan
		Data weighting: 200 API credits per symbol"
    symbol: Symbol ticker of mutual fund
        dp: Number of decimal places for floating values
        country: Filter by country name or alpha code
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/mutual_funds/world/sustainability"
    querystring = {'symbol': symbol, }
    if dp:
        querystring['dp'] = dp
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def performance(symbol: str, dp: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API request returns the detailed performance of a mutual fund, including trailing, annual, quarterly, and load-adjusted returns.
		
		Availability: Mega plan
		Data weighting: 200 API credits per symbol"
    symbol: Symbol ticker of mutual fund
        dp: Number of decimal places for floating values
        country: Filter by country name or alpha code
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/mutual_funds/world/performance"
    querystring = {'symbol': symbol, }
    if dp:
        querystring['dp'] = dp
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def summary(symbol: str='VFIAX', dp: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API request returns a brief summary of a mutual fund.
		
		Availability: Mega plan
		Data weighting: 200 API credits per symbol"
    symbol: Symbol ticker of mutual fund
        dp: Number of decimal places for floating values
        country: Filter by country name or alpha code
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/mutual_funds/world/summary"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    if dp:
        querystring['dp'] = dp
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mfs_family_list(country: str=None, fund_family: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API request returns a list of mutual funds families."
    country: Filter by country name or alpha code
        fund_family: Filter by investment company that manages the fund
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/mutual_funds/family"
    querystring = {}
    if country:
        querystring['country'] = country
    if fund_family:
        querystring['fund_family'] = fund_family
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mfs_list(apikey: str='demo', performance_rating: str=None, fund_type: str=None, outputsize: str=None, country: str=None, fund_family: str=None, symbol: str=None, risk_rating: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API request returns a list of mutual funds available at Twelve Data. Sorting is in descending order by total assets value. The list is updated daily.
		
		With Basic, Pro, and Ultra plans, only 50 records will be in response. Mega and Custom plans can access complete data on over 200,000 Mutual Funds."
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/mutual_funds/list"
    querystring = {}
    if apikey:
        querystring['apikey'] = apikey
    if performance_rating:
        querystring['performance_rating'] = performance_rating
    if fund_type:
        querystring['fund_type'] = fund_type
    if outputsize:
        querystring['outputsize'] = outputsize
    if country:
        querystring['country'] = country
    if fund_family:
        querystring['fund_family'] = fund_family
    if symbol:
        querystring['symbol'] = symbol
    if risk_rating:
        querystring['risk_rating'] = risk_rating
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_movers_etfs(country: str=None, outputsize: str=None, direction: str=None, dp: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the top gaining or losing ETFs today.
		
		Top gainers are ordered by the highest rate of price increase since the previous day's close. Top losers are ordered by the highest percentage of price decrease since the last day.
		
		Availability: Ultra plan
		Data weighting: 100 API credits per symbol"
    country: Country of the snapshot, applicable to non-currencies only

Takes country name or alpha code; default `USA`
        outputsize: Specifies the size of the snapshot

Can be in a range from `1` to `50`; default `30`
        direction: Specifies the direction of the snapshot `gainers` or `losers`

By `default` gainers
        dp: Specifies the number of decimal places for floating values

Should be in range `[0,11]` inclusive; default `5`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/market_movers/etf"
    querystring = {}
    if country:
        querystring['country'] = country
    if outputsize:
        querystring['outputsize'] = outputsize
    if direction:
        querystring['direction'] = direction
    if dp:
        querystring['dp'] = dp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_movers_stocks(country: str=None, direction: str=None, dp: str=None, outputsize: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the top gaining or losing stocks today.
		
		Top gainers are ordered by the highest rate of price increase since the previous day's close. Top losers are ordered by the highest percentage of price decrease since the last day.
		
		Availability: Ultra plan
		Data weighting: 100 API credits per symbol"
    country: Country of the snapshot, applicable to non-currencies only

Takes country name or alpha code; default `USA`
        direction: Specifies the direction of the snapshot `gainers` or `losers`

By `default` gainers
        dp: Specifies the number of decimal places for floating values

Should be in range `[0,11]` inclusive; default `5`
        outputsize: Specifies the size of the snapshot

Can be in a range from `1` to `50`; default `30`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/market_movers/stocks"
    querystring = {}
    if country:
        querystring['country'] = country
    if direction:
        querystring['direction'] = direction
    if dp:
        querystring['dp'] = dp
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stocks_list(type: str=None, symbol: str=None, format: str='json', country: str=None, exchange: str='NASDAQ', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API call returns an array of symbols available at Twelve Data API. This list is updated daily."
    type: Filter by country name or alpha code
        symbol: Filter by symbol
        format: Value can be `CSV` or `JSON`
Default `JSON`
        country: Filter by country name or alpha code
        exchange: Filter by exchange name or mic code
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/stocks"
    querystring = {}
    if type:
        querystring['type'] = type
    if symbol:
        querystring['symbol'] = symbol
    if format:
        querystring['format'] = format
    if country:
        querystring['country'] = country
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_movers_crypto(outputsize: str=None, direction: str=None, dp: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the top gaining or losing crypto today.
		
		Top gainers are ordered by the highest rate of price increase since the previous day's close. Top losers are ordered by the highest percentage of price decrease since the last day.
		
		Availability: Ultra plan
		Data weighting: 100 API credits per symbol"
    outputsize: Specifies the size of the snapshot

Can be in a range from `1` to `50`; default `30`
        direction: Specifies the direction of the snapshot `gainers` or `losers`

By `default` gainers
        dp: Specifies the number of decimal places for floating values

Should be in range `[0,11]` inclusive; default `5`
        country: Country of the snapshot, applicable to non-currencies only

Takes country name or alpha code; default `USA`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/market_movers/crypto"
    querystring = {}
    if outputsize:
        querystring['outputsize'] = outputsize
    if direction:
        querystring['direction'] = direction
    if dp:
        querystring['dp'] = dp
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_movers_forex(dp: str=None, direction: str=None, country: str=None, outputsize: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the top gaining or losing forex today.
		
		Top gainers are ordered by the highest rate of price increase since the previous day's close. Top losers are ordered by the highest percentage of price decrease since the last day.
		
		Availability: Ultra plan
		Data weighting: 100 API credits per symbol"
    dp: Specifies the number of decimal places for floating values

Should be in range `[0,11]` inclusive; default `5`
        direction: Specifies the direction of the snapshot `gainers` or `losers`

By `default` gainers
        country: Country of the snapshot, applicable to non-currencies only

Takes country name or alpha code; default `USA`
        outputsize: Specifies the size of the snapshot

Can be in a range from `1` to `50`; default `30`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/market_movers/forex"
    querystring = {}
    if dp:
        querystring['dp'] = dp
    if direction:
        querystring['direction'] = direction
    if country:
        querystring['country'] = country
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_data(symbol: str, country: str=None, dp: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API request returns a complete breakdown of a mutual fund, including summary, performance, risk, ratings, composition, purchase_info, and sustainability.
		
		Availability: Mega plan
		Data weighting: 1000 API credits per symbol"
    symbol: Symbol ticker of mutual fund
        country: Filter by country name or alpha code
        dp: Number of decimal places for floating values
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/mutual_funds/world"
    querystring = {'symbol': symbol, }
    if country:
        querystring['country'] = country
    if dp:
        querystring['dp'] = dp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_movers(outputsize: str=None, dp: str=None, direction: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of the top gaining or losing mutual funds today.
		
		Top gainers are ordered by the highest rate of price increase since the previous day's close. Top losers are ordered by the highest percentage of price decrease since the last day.
		
		Availability: Ultra plan
		Data weighting: 100 API credits per symbol"
    outputsize: Specifies the size of the snapshot

Can be in a range from `1` to `50`; default `30`
        dp: Specifies the number of decimal places for floating values

Should be in range `[0,11]` inclusive; default `5`
        direction: Specifies the direction of the snapshot `gainers` or `losers`

By `default` gainers
        country: Country of the snapshot, applicable to non-currencies only

Takes country name or alpha code; default `USA`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/market_movers/mutual_funds"
    querystring = {}
    if outputsize:
        querystring['outputsize'] = outputsize
    if dp:
        querystring['dp'] = dp
    if direction:
        querystring['direction'] = direction
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def risk(symbol: str, dp: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API request returns core metrics to measure the risk of investing in a mutual fund.
		
		Availability: Mega plan
		Data weighting: 200 API credits per symbol"
    symbol: Symbol ticker of mutual fund
        dp: Number of decimal places for floating values
        country: Filter by country name or alpha code
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/mutual_funds/world/risk"
    querystring = {'symbol': symbol, }
    if dp:
        querystring['dp'] = dp
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mfs_type_list(country: str=None, fund_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API request returns a list of mutual funds types."
    country: Filter by country name or alpha code
        fund_type: Filter by the type of fund
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/mutual_funds/type"
    querystring = {}
    if country:
        querystring['country'] = country
    if fund_type:
        querystring['fund_type'] = fund_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earliest_timestamp(symbol: str, interval: str, outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method returns the first available DateTime for a given instrument at the specific interval."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        outputsize: Default `30`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/earliest_timestamp"
    querystring = {'symbol': symbol, 'interval': interval, }
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbol_search(symbol: str='AA', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method helps to find the best matching symbol. It can be used as the base for custom lookups. The response is returned in descending order, with the most relevant instrument at the beginning."
    symbol: Symbol to search
        outputsize: Number of matches in response
Default `30`, Max `120`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/symbol_search"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def technical_indicators_interface(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API call return array of objects with available technical indicators. This endpoint might be used to build an abstract interface to make more convenient API calls from the application."
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/technical_indicators"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cryptocurrencies_list(symbol: str=None, currency_quote: str=None, currency_base: str='BTC', format: str='json', exchange: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API call return array of cryptocurrency pairs available at Twelve Data API. This list is daily updated."
    symbol: Filter by symbol
        currency_quote: Filter by currency quote
        currency_base: Filter by currency base
        format: Value can be `CSV` or `JSON`
Default `JSON`
        exchange: Filter by crypto exchange name
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/cryptocurrencies"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    if currency_quote:
        querystring['currency_quote'] = currency_quote
    if currency_base:
        querystring['currency_base'] = currency_base
    if format:
        querystring['format'] = format
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def etfs_list(exchange: str='Euronext', symbol: str=None, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API call return array of ETFs available at Twelve Data API. This list is daily updated."
    exchange: Filter by exchange name or mic code
        symbol: Filter by symbol
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/etf"
    querystring = {}
    if exchange:
        querystring['exchange'] = exchange
    if symbol:
        querystring['symbol'] = symbol
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forex_pairs_list(currency_base: str='EUR', symbol: str=None, format: str='json', currency_quote: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API call return array of forex pairs available at Twelve Data API. This list is daily updated."
    currency_base: Filter by currency base
        symbol: Filter by symbol
        format: Value can be `CSV` or `JSON`
Default `JSON`
        currency_quote: Filter by currency quote
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/forex_pairs"
    querystring = {}
    if currency_base:
        querystring['currency_base'] = currency_base
    if symbol:
        querystring['symbol'] = symbol
    if format:
        querystring['format'] = format
    if currency_quote:
        querystring['currency_quote'] = currency_quote
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def indices_list(exchange: str='NYSE', format: str='json', symbol: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API call return array of indices available at Twelve Data API. This list is daily updated."
    exchange: Filter by exchange name or mic code
        format: Value can be `CSV` or `JSON`
Default `JSON`
        symbol: Filter by symbol
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/indices"
    querystring = {}
    if exchange:
        querystring['exchange'] = exchange
    if format:
        querystring['format'] = format
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_conversion(symbol: str, amount: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API call returns real-time exchange rate and converted amount for currency pair. Works with forex and cryptocurrency."
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/currency_conversion"
    querystring = {'symbol': symbol, 'amount': amount, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange_rate(symbol: str, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API call returns real-time exchange rate for currency pair. Works with forex and cryptocurrency."
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/exchange_rate"
    querystring = {'symbol': symbol, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def key_executives(symbol: str, exchange: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns individuals at the highest level of management of an organization.
		
		Availability: Mega plan
		Data weighting: 1000 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/key_executives"
    querystring = {'symbol': symbol, }
    if exchange:
        querystring['exchange'] = exchange
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def options_chain(symbol: str, option_id: str=None, expiration_date: str=None, exchange: str=None, side: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a listing of all available options contracts for given security. It shows all listed puts, calls, their expiration, strike prices, and pricing information for a single underlying asset within a given maturity period.
		
		Availability: Ultra plan
		Data weighting: 50 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/options/chain"
    querystring = {'symbol': symbol, }
    if option_id:
        querystring['option_id'] = option_id
    if expiration_date:
        querystring['expiration_date'] = expiration_date
    if exchange:
        querystring['exchange'] = exchange
    if side:
        querystring['side'] = side
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statistics(symbol: str=None, exchange: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current overview of company’s main statistics including valuation metrics and financials.
		
		Availability: Ultra plan
		Data weighting: 50 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/statistics"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    if exchange:
        querystring['exchange'] = exchange
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ipo_calendar(end_date: str=None, symbol: str=None, country: str=None, start_date: str=None, exchange: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns past, today, or upcoming IPOs.
		
		Availability: Pro plan
		Data weighting: 40 API credits per symbol"
    
    """
    url = f"https://twelve-data1.p.rapidapi.com/ipo_calendar"
    querystring = {}
    if end_date:
        querystring['end_date'] = end_date
    if symbol:
        querystring['symbol'] = symbol
    if country:
        querystring['country'] = country
    if start_date:
        querystring['start_date'] = start_date
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coppock(interval: str, symbol: str, long_roc_period: int=14, format: str='json', outputsize: int=30, wma_period: int=10, series_type: str='close', short_roc_period: int=11, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Coppock Curve(COPPOCK) is usually used to detect long-term trend changes, typically on monthly charts."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/coppock"
    querystring = {'interval': interval, 'symbol': symbol, }
    if long_roc_period:
        querystring['long_roc_period'] = long_roc_period
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if wma_period:
        querystring['wma_period'] = wma_period
    if series_type:
        querystring['series_type'] = series_type
    if short_roc_period:
        querystring['short_roc_period'] = short_roc_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keltner(interval: str, symbol: str, atr_time_period: int=10, time_period: int=20, multiplier: int=2, format: str='json', series_type: str='close', ma_type: str='SMA', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Keltner Channels(KELTNER) is a volatility indicator used to spot trend changes and accelerations."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        ma_type: Supports: `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `KAMA`, `MAMA`, `T3MA`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/keltner"
    querystring = {'interval': interval, 'symbol': symbol, }
    if atr_time_period:
        querystring['atr_time_period'] = atr_time_period
    if time_period:
        querystring['time_period'] = time_period
    if multiplier:
        querystring['multiplier'] = multiplier
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    if ma_type:
        querystring['ma_type'] = ma_type
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dx(interval: str, symbol: str, time_period: int=14, outputsize: int=30, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Directional Movement Index(DX) identifies which direction the price is moving."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/dx"
    querystring = {'interval': interval, 'symbol': symbol, }
    if time_period:
        querystring['time_period'] = time_period
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ultosc(symbol: str, interval: str, time_period_1: int=7, time_period_2: int=14, format: str='json', time_period_3: int=28, outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ultimate Oscillator(ULTOSC) takes into account three different time periods to enhance the quality of overbought and oversold signals."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/ultosc"
    querystring = {'symbol': symbol, 'interval': interval, }
    if time_period_1:
        querystring['time_period_1'] = time_period_1
    if time_period_2:
        querystring['time_period_2'] = time_period_2
    if format:
        querystring['format'] = format
    if time_period_3:
        querystring['time_period_3'] = time_period_3
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wma(symbol: str, interval: str, format: str='json', outputsize: int=30, time_period: int=9, series_type: str='close', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Weighted Moving Average(WMA) smooths out price fluctuations, and puts more weight on recent data points and less on past."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/wma"
    querystring = {'symbol': symbol, 'interval': interval, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    if series_type:
        querystring['series_type'] = series_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trange(interval: str, symbol: str, outputsize: int=30, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "True Range(TRANGE) is usually used as the base when calculating other indicators. TRANGE determines the normal trading range of an asset."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/trange"
    querystring = {'interval': interval, 'symbol': symbol, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def macdext(symbol: str, interval: str, signal_ma_type: str='SMA', series_type: str='close', format: str='json', signal_period: int=9, slow_ma_type: str='SMA', fast_period: int=12, outputsize: int=30, fast_ma_type: str='SMA', slow_period: int=26, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Moving Average Convergence Divergence Extended(MACDEXT) gives greater control over MACD input parameters. MACDEXT has an unstable period ~ 100."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        signal_ma_type: Supports: `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `KAMA`, `MAMA`, `T3MA`
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        slow_ma_type: Supports: `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `KAMA`, `MAMA`, `T3MA`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        fast_ma_type: Supports: `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `KAMA`, `MAMA`, `T3MA`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/macdext"
    querystring = {'symbol': symbol, 'interval': interval, }
    if signal_ma_type:
        querystring['signal_ma_type'] = signal_ma_type
    if series_type:
        querystring['series_type'] = series_type
    if format:
        querystring['format'] = format
    if signal_period:
        querystring['signal_period'] = signal_period
    if slow_ma_type:
        querystring['slow_ma_type'] = slow_ma_type
    if fast_period:
        querystring['fast_period'] = fast_period
    if outputsize:
        querystring['outputsize'] = outputsize
    if fast_ma_type:
        querystring['fast_ma_type'] = fast_ma_type
    if slow_period:
        querystring['slow_period'] = slow_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def linearregslope(interval: str, symbol: str, format: str='json', time_period: int=9, series_type: str='close', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Linear Regression Slope(LINEARREGSLOPE) calculates the slope for the linear regression trendline for each data point."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/linearregslope"
    querystring = {'interval': interval, 'symbol': symbol, }
    if format:
        querystring['format'] = format
    if time_period:
        querystring['time_period'] = time_period
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hlc3(symbol: str, interval: str, format: str='json', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "High, Low, Close Average Values(HLC3) give alternative candlesticks patter. Every element is defined as follows: (high + low + close) / 3."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/hlc3"
    querystring = {'symbol': symbol, 'interval': interval, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def minmaxindex(symbol: str, interval: str, time_period: int=9, format: str='json', series_type: str='close', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Indexes of lowest and highest values over period(MINMAXINDEX)."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/minmaxindex"
    querystring = {'symbol': symbol, 'interval': interval, }
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def medprice(interval: str, symbol: str, format: str='json', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Median Price(MEDPRICE)."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/medprice"
    querystring = {'interval': interval, 'symbol': symbol, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def linearreg(interval: str, symbol: str, time_period: int=9, format: str='json', series_type: str='close', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Linear Regression(LINEARREG) is used to determine trend direction by a straight line."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/linearreg"
    querystring = {'interval': interval, 'symbol': symbol, }
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ht_trendline(interval: str, symbol: str, series_type: str='close', outputsize: int=30, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hilbert Transform Instantaneous Trendline(HT_TRENDLINE) comes from the concept of Digital Signal Processing (DSP). It creates complex signals from the simple chart data. You can read more about it in the Rocket Science for Traders book by John F. Ehlers."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/ht_trendline"
    querystring = {'interval': interval, 'symbol': symbol, }
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def avgprice(interval: str, symbol: str, outputsize: int=30, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Average Price(AVGPRICE) uses the formula: (open + high + low + close) / 4."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/avgprice"
    querystring = {'interval': interval, 'symbol': symbol, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ht_trendmode(symbol: str, interval: str, series_type: str='close', format: str='json', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hilbert Transform Trend vs. Cycle Mode(HT_TRENDMODE) is part of Hilbert Transforms concepts. You can read more about it in the Rocket Science for Traders book by John F. Ehlers."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/ht_trendmode"
    querystring = {'symbol': symbol, 'interval': interval, }
    if series_type:
        querystring['series_type'] = series_type
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ma(interval: str, symbol: str, time_period: int=9, outputsize: int=30, ma_type: str='SMA', series_type: str='close', format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Moving Average(MA) is used to smooth out price fluctuations and get rid of market noise."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        ma_type: Supports: `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `KAMA`, `MAMA`, `T3MA`
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/ma"
    querystring = {'interval': interval, 'symbol': symbol, }
    if time_period:
        querystring['time_period'] = time_period
    if outputsize:
        querystring['outputsize'] = outputsize
    if ma_type:
        querystring['ma_type'] = ma_type
    if series_type:
        querystring['series_type'] = series_type
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exp(symbol: str, interval: str, series_type: str='close', outputsize: int=30, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Exponential(EXP) transforms input data with the mathematical exponent function."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/exp"
    querystring = {'symbol': symbol, 'interval': interval, }
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wclprice(interval: str, symbol: str, outputsize: int=30, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Weighted Close Price(WCLPRICE) is usually used as the base for other indicators for smoothness. Formula: (high + low + close * 2) / 4."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/wclprice"
    querystring = {'interval': interval, 'symbol': symbol, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stochf(interval: str, symbol: str, fast_dma_type: str='SMA', format: str='json', fast_k_period: int=14, fast_d_period: int=3, outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stochastic Fast(STOCHF) is more sensitive to price changes; therefore, it changes direction more quickly."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        fast_dma_type: Supports: `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `KAMA`, `MAMA`, `T3MA`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/stochf"
    querystring = {'interval': interval, 'symbol': symbol, }
    if fast_dma_type:
        querystring['fast_dma_type'] = fast_dma_type
    if format:
        querystring['format'] = format
    if fast_k_period:
        querystring['fast_k_period'] = fast_k_period
    if fast_d_period:
        querystring['fast_d_period'] = fast_d_period
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kama(symbol: str, interval: str, outputsize: int=30, time_period: int=9, format: str='json', series_type: str='close', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Kaufman's Adaptive Moving Average(KAMA) is a type of Moving Average(MA) that incorporates market noise and volatility."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/kama"
    querystring = {'symbol': symbol, 'interval': interval, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def adosc(interval: str, symbol: str, slow_period: int=26, fast_period: int=12, format: str='json', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Chaikin A/D Oscillator(ADOSC) is an indicator, which finds the relationship between increasing and decreasing volume with price fluctuations. The Chaikin Oscillator measures the momentum of the Accumulation/Distribution Line(ADL) using two Exponential Moving Averages of varying length to the line(MACD)."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/adosc"
    querystring = {'interval': interval, 'symbol': symbol, }
    if slow_period:
        querystring['slow_period'] = slow_period
    if fast_period:
        querystring['fast_period'] = fast_period
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def min(symbol: str, interval: str, outputsize: int=30, series_type: str='close', time_period: int=9, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lowest value over period(MIN)."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/min"
    querystring = {'symbol': symbol, 'interval': interval, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if series_type:
        querystring['series_type'] = series_type
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vwap(interval: str, symbol: str, format: str='json', outputsize: int=30, sd_time_period: int=0, sd: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Volume Weighted Average Price(VWAP) is commonly used as a trading benchmark that gives an average price at which the instrument has been trading during the day."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/vwap"
    querystring = {'interval': interval, 'symbol': symbol, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if sd_time_period:
        querystring['sd_time_period'] = sd_time_period
    if sd:
        querystring['sd'] = sd
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def plus_dm(symbol: str, interval: str, outputsize: int=30, time_period: int=9, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Plus Directional Movement(PLUS_DM) is calculated as High - Previous High."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/plus_dm"
    querystring = {'symbol': symbol, 'interval': interval, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cmo(symbol: str, interval: str, outputsize: int=30, series_type: str='close', format: str='json', time_period: int=9, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Chande Momentum Oscillator(CMO) is used to show overbought and oversold conditions."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/cmo"
    querystring = {'symbol': symbol, 'interval': interval, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if series_type:
        querystring['series_type'] = series_type
    if format:
        querystring['format'] = format
    if time_period:
        querystring['time_period'] = time_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def willr(interval: str, symbol: str, time_period: int=9, format: str='json', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Williams %R(WILLR) calculates overbought and oversold levels. It can also be used to find entry and exit signals."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/willr"
    querystring = {'interval': interval, 'symbol': symbol, }
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mom(interval: str, symbol: str, series_type: str='close', format: str='json', outputsize: int=30, time_period: int=9, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Momentum(MOM) compares the current price with the previous price N timeperiods ago."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/mom"
    querystring = {'interval': interval, 'symbol': symbol, }
    if series_type:
        querystring['series_type'] = series_type
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def linearregangle(interval: str, symbol: str, time_period: int=9, format: str='json', outputsize: int=30, series_type: str='close', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Linear Regression Angle(LINEARREGANGLE) calculates the angle of the linear regression trendline."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/linearregangle"
    querystring = {'interval': interval, 'symbol': symbol, }
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if series_type:
        querystring['series_type'] = series_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rocr(interval: str, symbol: str, format: str='json', time_period: int=9, series_type: str='close', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rate of change ratio(ROCR) calculates the ratio between the current price and price n timeperiods ago. Formula: (price / prevPrice)."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/rocr"
    querystring = {'interval': interval, 'symbol': symbol, }
    if format:
        querystring['format'] = format
    if time_period:
        querystring['time_period'] = time_period
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ema(symbol: str, interval: str, series_type: str='close', format: str='json', outputsize: int=30, time_period: int=9, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Exponential Moving Average(EMA) places greater importance on recent data points than the normal Moving Average(MA)."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/ema"
    querystring = {'symbol': symbol, 'interval': interval, }
    if series_type:
        querystring['series_type'] = series_type
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sub(interval: str, symbol: str, outputsize: int=30, format: str='json', series_type_1: str='open', series_type_2: str='close', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Subtraction of values of two specified time series."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type_1: Supports: `open`, `high`, `low`, `close`
        series_type_2: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/sub"
    querystring = {'interval': interval, 'symbol': symbol, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    if series_type_1:
        querystring['series_type_1'] = series_type_1
    if series_type_2:
        querystring['series_type_2'] = series_type_2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def apo(interval: str, symbol: str, slow_period: int=26, fast_period: int=12, format: str='json', ma_type: str='SMA', series_type: str='close', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Absolute Price Oscillator(APO) calculates the difference between two price moving averages."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        ma_type: Supports: `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `KAMA`, `MAMA`, `T3MA`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/apo"
    querystring = {'interval': interval, 'symbol': symbol, }
    if slow_period:
        querystring['slow_period'] = slow_period
    if fast_period:
        querystring['fast_period'] = fast_period
    if format:
        querystring['format'] = format
    if ma_type:
        querystring['ma_type'] = ma_type
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rocr100(interval: str, symbol: str, outputsize: int=30, time_period: int=9, format: str='json', series_type: str='close', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rate of change ratio 100 scale(ROCR100) calculates the ratio with 100 scale between the current price and price n timeperiods ago. Formula: (price / prevPrice) * 100."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/rocr100"
    querystring = {'interval': interval, 'symbol': symbol, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def typprice(interval: str, symbol: str, format: str='json', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Typical Price(TYPPRICE)."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/typprice"
    querystring = {'interval': interval, 'symbol': symbol, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sma(interval: str, symbol: str, time_period: int=9, outputsize: int=30, format: str='json', series_type: str='close', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simple Moving Average(SMA) is an arithmetic moving average calculated by adding the latest closing prices and then dividing them by the number of time periods."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/sma"
    querystring = {'interval': interval, 'symbol': symbol, }
    if time_period:
        querystring['time_period'] = time_period
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def var(symbol: str, interval: str, time_period: int=9, format: str='json', series_type: str='close', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Variance(VAR) calculates the spread between data points to determine how far they from the mean."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/var"
    querystring = {'symbol': symbol, 'interval': interval, }
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stddev(interval: str, symbol: str, series_type: str='close', sd: int=2, time_period: int=9, format: str='json', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Standard Deviation(STDDEV) is used to measure volatility. This might be important when assessing risks."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/stddev"
    querystring = {'interval': interval, 'symbol': symbol, }
    if series_type:
        querystring['series_type'] = series_type
    if sd:
        querystring['sd'] = sd
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cci(symbol: str, interval: str, outputsize: int=30, time_period: int=20, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Commodity Channel Index(CCI) is a universal indicator that can help to identify new trends and assess current critical conditions."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/cci"
    querystring = {'symbol': symbol, 'interval': interval, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ht_phasor(symbol: str, interval: str, format: str='json', series_type: str='close', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hilbert Transform Phasor Components(HT_PHASOR) is part of Hilbert Transforms concepts. You can read more about it in the Rocket Science for Traders book by John F. Ehlers."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/ht_phasor"
    querystring = {'symbol': symbol, 'interval': interval, }
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def adx(symbol: str, interval: str, format: str='json', outputsize: int=30, time_period: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Average Directional Index(ADX) is used to decide if the price trend is strong."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/adx"
    querystring = {'symbol': symbol, 'interval': interval, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sarext(symbol: str, interval: str, offset_on_reverse: int=0, acceleration_max_short: int=None, start_value: int=0, format: str='json', acceleration_limit_long: int=None, acceleration_limit_short: int=None, acceleration_short: int=None, outputsize: int=30, acceleration_max_long: int=None, acceleration_long: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parabolic SAR Extended(SAREXT) inherits the idea of classic SAR indicator but adds more flexibility in setting parameters."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/sarext"
    querystring = {'symbol': symbol, 'interval': interval, }
    if offset_on_reverse:
        querystring['offset_on_reverse'] = offset_on_reverse
    if acceleration_max_short:
        querystring['acceleration_max_short'] = acceleration_max_short
    if start_value:
        querystring['start_value'] = start_value
    if format:
        querystring['format'] = format
    if acceleration_limit_long:
        querystring['acceleration_limit_long'] = acceleration_limit_long
    if acceleration_limit_short:
        querystring['acceleration_limit_short'] = acceleration_limit_short
    if acceleration_short:
        querystring['acceleration_short'] = acceleration_short
    if outputsize:
        querystring['outputsize'] = outputsize
    if acceleration_max_long:
        querystring['acceleration_max_long'] = acceleration_max_long
    if acceleration_long:
        querystring['acceleration_long'] = acceleration_long
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def log10(symbol: str, interval: str, format: str='json', series_type: str='close', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Logarithm to base 10(LOG10) transforms all data points with logarithm to base 10."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/log10"
    querystring = {'symbol': symbol, 'interval': interval, }
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crsi(interval: str, symbol: str, percent_rank_period: int=100, outputsize: int=30, series_type: str='close', format: str='json', rsi_period: int=3, up_down_length: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ConnorsRSI(CRSI) is used to show the oversold and overbought levels of the RSI values."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/crsi"
    querystring = {'interval': interval, 'symbol': symbol, }
    if percent_rank_period:
        querystring['percent_rank_period'] = percent_rank_period
    if outputsize:
        querystring['outputsize'] = outputsize
    if series_type:
        querystring['series_type'] = series_type
    if format:
        querystring['format'] = format
    if rsi_period:
        querystring['rsi_period'] = rsi_period
    if up_down_length:
        querystring['up_down_length'] = up_down_length
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dpo(interval: str, symbol: str, outputsize: int=30, time_period: int=21, centered: bool=None, series_type: str='close', format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Detrended Price Oscillator(DPO) is used to separate price from the trend, in order to more clearly identify the length of cycles."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/dpo"
    querystring = {'interval': interval, 'symbol': symbol, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    if centered:
        querystring['centered'] = centered
    if series_type:
        querystring['series_type'] = series_type
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def macd(interval: str, symbol: str, signal_period: int=9, outputsize: int=30, series_type: str='close', fast_period: int=12, slow_period: int=26, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Moving Average Convergence Divergence(MACD) is a trend following momentum indicator, which works by subtracting the longer moving average from the shorter one. MACD has an unstable period ~ 100."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/macd"
    querystring = {'interval': interval, 'symbol': symbol, }
    if signal_period:
        querystring['signal_period'] = signal_period
    if outputsize:
        querystring['outputsize'] = outputsize
    if series_type:
        querystring['series_type'] = series_type
    if fast_period:
        querystring['fast_period'] = fast_period
    if slow_period:
        querystring['slow_period'] = slow_period
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def linearregintercept(symbol: str, interval: str, series_type: str='close', time_period: int=9, format: str='json', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Linear Regression Intercept(LINEARREGINTERCEPT) calculates the intercept for the linear regression trendline for each data point."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/linearregintercept"
    querystring = {'symbol': symbol, 'interval': interval, }
    if series_type:
        querystring['series_type'] = series_type
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def minindex(interval: str, symbol: str, format: str='json', outputsize: int=30, series_type: str='close', time_period: int=9, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Index of lowest value over period(MININDEX)."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/minindex"
    querystring = {'interval': interval, 'symbol': symbol, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if series_type:
        querystring['series_type'] = series_type
    if time_period:
        querystring['time_period'] = time_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def midpoint(interval: str, symbol: str, format: str='json', outputsize: int=30, time_period: int=9, series_type: str='close', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "MidPoint over period(MIDPOINT) is calculated as (highest value + lowest value) / 2."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/midpoint"
    querystring = {'interval': interval, 'symbol': symbol, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    if series_type:
        querystring['series_type'] = series_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def add(interval: str, symbol: str, outputsize: int=30, series_type_2: str='close', series_type_1: str='open', format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Composite of values of two specified time series."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type_2: Supports: `open`, `high`, `low`, `close`
        series_type_1: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/add"
    querystring = {'interval': interval, 'symbol': symbol, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if series_type_2:
        querystring['series_type_2'] = series_type_2
    if series_type_1:
        querystring['series_type_1'] = series_type_1
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stoch(symbol: str, interval: str, slow_k_period: int=1, slow_kma_type: str='SMA', slow_dma_type: str='SMA', fast_k_period: int=14, format: str='json', slow_d_period: int=3, outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stochastic Oscillator(STOCH) is used to decide if the price trend is strong."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        slow_kma_type: Supports: `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `KAMA`, `MAMA`, `T3MA`
        slow_dma_type: Supports: `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `KAMA`, `MAMA`, `T3MA`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/stoch"
    querystring = {'symbol': symbol, 'interval': interval, }
    if slow_k_period:
        querystring['slow_k_period'] = slow_k_period
    if slow_kma_type:
        querystring['slow_kma_type'] = slow_kma_type
    if slow_dma_type:
        querystring['slow_dma_type'] = slow_dma_type
    if fast_k_period:
        querystring['fast_k_period'] = fast_k_period
    if format:
        querystring['format'] = format
    if slow_d_period:
        querystring['slow_d_period'] = slow_d_period
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ht_dcphase(symbol: str, interval: str, series_type: str='close', format: str='json', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hilbert Transform Dominant Cycle Phase(HT_DCPHASE) is part of Hilbert Transforms concepts. You can read more about it in the Rocket Science for Traders book by John F. Ehlers."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/ht_dcphase"
    querystring = {'symbol': symbol, 'interval': interval, }
    if series_type:
        querystring['series_type'] = series_type
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchanges(name: str=None, type: str=None, format: str='json', code: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API call return array of stock, ETF or index exchanges available at Twelve Data API. This list is daily updated."
    name: Filter by exchange name
        type: Value can be `stock`, `etf` or `index`
Default `stock`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        code: Filter by exchange mic code
        country: Filter by country name or alpha code
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/exchanges"
    querystring = {}
    if name:
        querystring['name'] = name
    if type:
        querystring['type'] = type
    if format:
        querystring['format'] = format
    if code:
        querystring['code'] = code
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sqrt(interval: str, symbol: str, outputsize: int=30, series_type: str='close', format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Square Root(SQRT) transforms input data with square root."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/sqrt"
    querystring = {'interval': interval, 'symbol': symbol, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if series_type:
        querystring['series_type'] = series_type
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trima(symbol: str, interval: str, format: str='json', outputsize: int=30, series_type: str='close', time_period: int=9, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Triangular Moving Average(TRIMA) smooths out price fluctuations, but places more weight on the prices in middle of the time period."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/trima"
    querystring = {'symbol': symbol, 'interval': interval, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if series_type:
        querystring['series_type'] = series_type
    if time_period:
        querystring['time_period'] = time_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tsf(symbol: str, interval: str, format: str='json', series_type: str='close', time_period: int=9, outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Time Series Forecast(TSF) calculates trend based on the last points of multiple regression trendlines."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/tsf"
    querystring = {'symbol': symbol, 'interval': interval, }
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    if time_period:
        querystring['time_period'] = time_period
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def adxr(interval: str, symbol: str, format: str='json', outputsize: int=30, time_period: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Average Directional Movement Index Rating(ADXR) is a smoothed version of the ADX indicator. ADXR quantifies momentum change in the ADX."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/adxr"
    querystring = {'interval': interval, 'symbol': symbol, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ht_dcperiod(symbol: str, interval: str, format: str='json', series_type: str='close', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hilbert Transform Dominant Cycle Period(HT_DCPERIOD) is part of Hilbert Transforms concepts. You can read more about it in the Rocket Science for Traders book by John F. Ehlers."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/ht_dcperiod"
    querystring = {'symbol': symbol, 'interval': interval, }
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def obv(symbol: str, interval: str, format: str='json', outputsize: int=30, series_type: str='close', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "On Balance Volume(OBV) is a momentum indicator, which uses volume flow to forecast upcoming price changes."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/obv"
    querystring = {'symbol': symbol, 'interval': interval, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if series_type:
        querystring['series_type'] = series_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sum(interval: str, symbol: str, time_period: int=9, format: str='json', series_type: str='close', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Summation of values of two specified time series."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/sum"
    querystring = {'interval': interval, 'symbol': symbol, }
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ichimoku(symbol: str, interval: str, base_line_period: int=26, format: str='json', leading_span_b_period: int=52, lagging_span_period: int=26, include_ahead_span_period: bool=None, conversion_line_period: int=9, outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Ichimoku Kinkō Hyō(ICHIMOKU) is a group of technical indicators that shows trend direction, momentum, and support & resistance levels. Overall it tends to improve the accuracy of forecasts."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/ichimoku"
    querystring = {'symbol': symbol, 'interval': interval, }
    if base_line_period:
        querystring['base_line_period'] = base_line_period
    if format:
        querystring['format'] = format
    if leading_span_b_period:
        querystring['leading_span_b_period'] = leading_span_b_period
    if lagging_span_period:
        querystring['lagging_span_period'] = lagging_span_period
    if include_ahead_span_period:
        querystring['include_ahead_span_period'] = include_ahead_span_period
    if conversion_line_period:
        querystring['conversion_line_period'] = conversion_line_period
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ad(interval: str, symbol: str, outputsize: int=30, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Chaikin A/D Line(AD) calculates the Advance/Decline of an asset. This indicator belongs to the group of Volume Indicators."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/ad"
    querystring = {'interval': interval, 'symbol': symbol, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def minus_di(symbol: str, interval: str, outputsize: int=30, time_period: int=9, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Minus Directional Indicator(MINUS_DI) is a component of the Average Directional Index(ADX), and it measures the existence of downtrend."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/minus_di"
    querystring = {'symbol': symbol, 'interval': interval, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mcginley_dynamic(symbol: str, interval: str, outputsize: int=30, format: str='json', time_period: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "McGinley Dynamic(MCGINLEY_DYNAMIC) keeps all the benefits from the moving averages but adds an adjustment to market speed."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/mcginley_dynamic"
    querystring = {'symbol': symbol, 'interval': interval, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    if time_period:
        querystring['time_period'] = time_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dema(interval: str, symbol: str, outputsize: int=30, time_period: int=9, series_type: str='close', format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Double Exponential Moving Average(DEMA) is used to eliminate lag. It does this by taking two Exponential Moving Averages(EMA)."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/dema"
    querystring = {'interval': interval, 'symbol': symbol, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    if series_type:
        querystring['series_type'] = series_type
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def avg(interval: str, symbol: str, outputsize: int=30, time_period: int=9, format: str='json', series_type: str='close', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Average(AVG) calculates the average value of series in a given time period. Widely used to calculate the 9-day average volume."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`, `volume`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/avg"
    querystring = {'interval': interval, 'symbol': symbol, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kst(interval: str, symbol: str, sma_period_3: int=10, roc_period_2: int=15, sma_period_4: int=15, format: str='json', sma_period_1: int=10, roc_period_3: int=20, outputsize: int=30, roc_period_4: int=30, roc_period_1: int=10, sma_period_2: int=10, signal_period: int=9, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Know Sure Thing(KST) calculates price momentum for four distinct price cycles(ROC)."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/kst"
    querystring = {'interval': interval, 'symbol': symbol, }
    if sma_period_3:
        querystring['sma_period_3'] = sma_period_3
    if roc_period_2:
        querystring['roc_period_2'] = roc_period_2
    if sma_period_4:
        querystring['sma_period_4'] = sma_period_4
    if format:
        querystring['format'] = format
    if sma_period_1:
        querystring['sma_period_1'] = sma_period_1
    if roc_period_3:
        querystring['roc_period_3'] = roc_period_3
    if outputsize:
        querystring['outputsize'] = outputsize
    if roc_period_4:
        querystring['roc_period_4'] = roc_period_4
    if roc_period_1:
        querystring['roc_period_1'] = roc_period_1
    if sma_period_2:
        querystring['sma_period_2'] = sma_period_2
    if signal_period:
        querystring['signal_period'] = signal_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aroonosc(interval: str, symbol: str, format: str='json', outputsize: int=30, time_period: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Aroon Oscillator(AROONOSC) uses classic Aroon(Aroon Up and Aroon down) to measure the strength of persisting trends and whether they will continue."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/aroonosc"
    querystring = {'interval': interval, 'symbol': symbol, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def correl(symbol: str, interval: str, series_type_2: str='close', outputsize: int=30, time_period: int=9, format: str='json', series_type_1: str='open', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Measures linear correlation between two time series. Takes values in the range from -1 to 1, where -1 is total negative correlation, 0 is no correlation, and 1 is total positive correlation. "
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        series_type_2: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type_1: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/correl"
    querystring = {'symbol': symbol, 'interval': interval, }
    if series_type_2:
        querystring['series_type_2'] = series_type_2
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    if series_type_1:
        querystring['series_type_1'] = series_type_1
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ht_sine(interval: str, symbol: str, series_type: str='close', format: str='json', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hilbert Transform SineWave(HT_SINE) is part of Hilbert Transforms concepts. You can read more about it in the Rocket Science for Traders book by John F. Ehlers."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/ht_sine"
    querystring = {'interval': interval, 'symbol': symbol, }
    if series_type:
        querystring['series_type'] = series_type
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bbands(symbol: str, interval: str, sd: int=2, series_type: str='close', ma_type: str='SMA', time_period: int=20, outputsize: int=30, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bollinger Bands®(BBANDS) are volatility bands located above and below a moving average. The volatility size parameter depends on standard deviation."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        series_type: Supports: `open`, `high`, `low`, `close`
        ma_type: Supports: `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `KAMA`, `MAMA`, `T3MA`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/bbands"
    querystring = {'symbol': symbol, 'interval': interval, }
    if sd:
        querystring['sd'] = sd
    if series_type:
        querystring['series_type'] = series_type
    if ma_type:
        querystring['ma_type'] = ma_type
    if time_period:
        querystring['time_period'] = time_period
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def natr(symbol: str, interval: str, outputsize: int=30, time_period: int=14, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Normalized Average True Range(NATR) is used to compare and analyze across different price levels due to its normalized quality, which might be more effective than the original ATR."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/natr"
    querystring = {'symbol': symbol, 'interval': interval, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aroon(symbol: str, interval: str, format: str='json', time_period: int=14, outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Aroon Indicator(AROON) is used to identify if the price is trending. It can also spot the beginning of a new trend and its strength."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/aroon"
    querystring = {'symbol': symbol, 'interval': interval, }
    if format:
        querystring['format'] = format
    if time_period:
        querystring['time_period'] = time_period
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def floor(symbol: str, interval: str, outputsize: int=30, series_type: str='close', format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Vector FLOOR(FLOOR) transforms input data with the mathematical floor function."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/floor"
    querystring = {'symbol': symbol, 'interval': interval, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if series_type:
        querystring['series_type'] = series_type
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rsi(interval: str, symbol: str, format: str='json', time_period: int=14, series_type: str='close', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Relative Strength Index(RSI) is a momentum indicator, which calculates the magnitude of a price change to assess the overbought and oversold conditions in the price of an asset."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/rsi"
    querystring = {'interval': interval, 'symbol': symbol, }
    if format:
        querystring['format'] = format
    if time_period:
        querystring['time_period'] = time_period
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def percent_b(interval: str, symbol: str, ma_type: str='SMA', sd: int=2, series_type: str='close', time_period: int=20, outputsize: int=30, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "%B Indicator(PERCENT_B) measures the position of an asset price relative to upper and lower Bollinger Bands."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        ma_type: Supports: `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `KAMA`, `MAMA`, `T3MA`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/percent_b"
    querystring = {'interval': interval, 'symbol': symbol, }
    if ma_type:
        querystring['ma_type'] = ma_type
    if sd:
        querystring['sd'] = sd
    if series_type:
        querystring['series_type'] = series_type
    if time_period:
        querystring['time_period'] = time_period
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stochrsi(symbol: str, interval: str, fast_dma_type: str='SMA', series_type: str='close', fast_d_period: int=3, outputsize: int=30, fast_k_period: int=3, format: str='json', time_period: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stochastic RSI(STOCHRSI) as an independent indicator takes advantage of the STOCH and RSI indicators. It is used to determine overbought and oversold levels, as well as current market trends for an asset."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        fast_dma_type: Supports: `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `KAMA`, `MAMA`, `T3MA`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/stochrsi"
    querystring = {'symbol': symbol, 'interval': interval, }
    if fast_dma_type:
        querystring['fast_dma_type'] = fast_dma_type
    if series_type:
        querystring['series_type'] = series_type
    if fast_d_period:
        querystring['fast_d_period'] = fast_d_period
    if outputsize:
        querystring['outputsize'] = outputsize
    if fast_k_period:
        querystring['fast_k_period'] = fast_k_period
    if format:
        querystring['format'] = format
    if time_period:
        querystring['time_period'] = time_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def max(symbol: str, interval: str, time_period: int=9, outputsize: int=30, series_type: str='close', format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Highest value over period(MAX)."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/max"
    querystring = {'symbol': symbol, 'interval': interval, }
    if time_period:
        querystring['time_period'] = time_period
    if outputsize:
        querystring['outputsize'] = outputsize
    if series_type:
        querystring['series_type'] = series_type
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def minus_dm(symbol: str, interval: str, time_period: int=9, format: str='json', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Minus Directional Movement(MINUS_DM) is calculated as Previous Low - Low."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/minus_dm"
    querystring = {'symbol': symbol, 'interval': interval, }
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def div(interval: str, symbol: str, series_type_1: str='open', format: str='json', outputsize: int=30, series_type_2: str='close', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Division of values of two specified time series."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        series_type_1: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type_2: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/div"
    querystring = {'interval': interval, 'symbol': symbol, }
    if series_type_1:
        querystring['series_type_1'] = series_type_1
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if series_type_2:
        querystring['series_type_2'] = series_type_2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def roc(interval: str, symbol: str, series_type: str='close', outputsize: int=30, format: str='json', time_period: int=9, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rate of change(ROC) calculates the rate of change between the current price and price n timeperiods ago. Formula: ((price / prevPrice) - 1) * 100."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/roc"
    querystring = {'interval': interval, 'symbol': symbol, }
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    if time_period:
        querystring['time_period'] = time_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rocp(interval: str, symbol: str, outputsize: int=30, time_period: int=9, format: str='json', series_type: str='close', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rate of change percentage(ROCP) calculates the rate of change in % between the current price and price n timeperiods ago. Formula: (price - prevPrice) / prevPrice."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/rocp"
    querystring = {'interval': interval, 'symbol': symbol, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bop(symbol: str, interval: str, format: str='json', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Balance of Power(BOP) measures the relative strength between buyers and sellers by assessing the ability of move price to an extreme level."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/bop"
    querystring = {'symbol': symbol, 'interval': interval, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mfi(interval: str, symbol: str, time_period: int=14, format: str='json', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Money Flow Index(MFI) is used to identify overbought and oversold levels in an asset. In some cases, it can be used to detect divergences, which might be a sign of upcoming trend changes."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/mfi"
    querystring = {'interval': interval, 'symbol': symbol, }
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mama(interval: str, symbol: str, format: str='json', slow_limit: int=None, series_type: str='close', fast_limit: int=None, outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "MESA Adaptive Moving Average(MAMA) adapts to price fluctuations based on the rate of change of the Hilbert Transform Discriminator. More about MAMA can be read [here](http://www.mesasoftware.com/papers/MAMA.pdf)."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/mama"
    querystring = {'interval': interval, 'symbol': symbol, }
    if format:
        querystring['format'] = format
    if slow_limit:
        querystring['slow_limit'] = slow_limit
    if series_type:
        querystring['series_type'] = series_type
    if fast_limit:
        querystring['fast_limit'] = fast_limit
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def beta(interval: str, symbol: str, format: str='json', series_type_2: str='close', series_type_1: str='open', time_period: int=9, outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Statistic Beta function."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type_2: Supports: `open`, `high`, `low`, `close`
        series_type_1: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/beta"
    querystring = {'interval': interval, 'symbol': symbol, }
    if format:
        querystring['format'] = format
    if series_type_2:
        querystring['series_type_2'] = series_type_2
    if series_type_1:
        querystring['series_type_1'] = series_type_1
    if time_period:
        querystring['time_period'] = time_period
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def atr(interval: str, symbol: str, time_period: int=14, outputsize: int=30, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Average True Range(ATR) is used to measure market volatility by decomposing all asset prices over a specified time period."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/atr"
    querystring = {'interval': interval, 'symbol': symbol, }
    if time_period:
        querystring['time_period'] = time_period
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def heikinashicandles(interval: str, symbol: str, format: str='json', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Heikin-Ashi Candles(HEIKINASHICANDLES) translated from Japanese means "average bar". It can be used to detect market trends and predict future price fluctuations."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/heikinashicandles"
    querystring = {'interval': interval, 'symbol': symbol, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ceil(interval: str, symbol: str, format: str='json', series_type: str='close', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Vector CEIL(CEIL) transforms input data with the mathematical ceil function."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/ceil"
    querystring = {'interval': interval, 'symbol': symbol, }
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mult(symbol: str, interval: str, series_type_2: str='close', format: str='json', series_type_1: str='open', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Multiplication of values of two specified time series."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        series_type_2: Supports: `open`, `high`, `low`, `close`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type_1: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/mult"
    querystring = {'symbol': symbol, 'interval': interval, }
    if series_type_2:
        querystring['series_type_2'] = series_type_2
    if format:
        querystring['format'] = format
    if series_type_1:
        querystring['series_type_1'] = series_type_1
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def midprice(interval: str, symbol: str, outputsize: int=30, time_period: int=9, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Midpoint Price over period(MIDPRICE) is calculated as (highest high + lowest low) / 2."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/midprice"
    querystring = {'interval': interval, 'symbol': symbol, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def plus_di(interval: str, symbol: str, time_period: int=9, outputsize: int=30, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Plus Directional Indicator(PLUS_DI) is a component of the Average Directional Index(ADX), and it measures the existence of uptrend."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/plus_di"
    querystring = {'interval': interval, 'symbol': symbol, }
    if time_period:
        querystring['time_period'] = time_period
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def supertrend(symbol: str, interval: str, period: int=10, format: str='json', multiplier: int=3, outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "SuperTrend Indicator(SUPERTREND) is mostly used on intraday timeframes to detect the price upward or downward direction in the trending market."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/supertrend"
    querystring = {'symbol': symbol, 'interval': interval, }
    if period:
        querystring['period'] = period
    if format:
        querystring['format'] = format
    if multiplier:
        querystring['multiplier'] = multiplier
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def maxindex(interval: str, symbol: str, time_period: int=9, series_type: str='close', outputsize: int=30, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Index of highest value over period(MAXINDEX)."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/maxindex"
    querystring = {'interval': interval, 'symbol': symbol, }
    if time_period:
        querystring['time_period'] = time_period
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def minmax(interval: str, symbol: str, series_type: str='close', outputsize: int=30, format: str='json', time_period: int=9, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lowest and highest values over period(MINMAX)."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/minmax"
    querystring = {'interval': interval, 'symbol': symbol, }
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    if time_period:
        querystring['time_period'] = time_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tema(interval: str, symbol: str, format: str='json', outputsize: int=30, time_period: int=9, series_type: str='close', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Triple Exponential Moving Average(TEMA) smooths out price fluctuations, making it more trend detection and more transparent without the lag."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        series_type: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/tema"
    querystring = {'interval': interval, 'symbol': symbol, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    if series_type:
        querystring['series_type'] = series_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sar(symbol: str, interval: str, maximum: int=None, outputsize: int=30, format: str='json', acceleration: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parabolic SAR(SAR) is used to identify and spot upcoming asset momentum."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/sar"
    querystring = {'symbol': symbol, 'interval': interval, }
    if maximum:
        querystring['maximum'] = maximum
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    if acceleration:
        querystring['acceleration'] = acceleration
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def t3ma(symbol: str, interval: str, format: str='json', v_factor: int=None, series_type: str='close', outputsize: int=30, time_period: int=9, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Triple Exponential Moving Average(T3MA) makes better smoothing of moving average than the classical TEMA indicator by extending the lookback period and applying other enhanced parameters."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/t3ma"
    querystring = {'symbol': symbol, 'interval': interval, }
    if format:
        querystring['format'] = format
    if v_factor:
        querystring['v_factor'] = v_factor
    if series_type:
        querystring['series_type'] = series_type
    if outputsize:
        querystring['outputsize'] = outputsize
    if time_period:
        querystring['time_period'] = time_period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crypto_exchanges(format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API call return array of cryptocurrency exchanges available at Twelve Data API. This list is daily updated."
    format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/cryptocurrency_exchanges"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_series(symbol: str, interval: str, outputsize: int=30, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API call returns meta and time series for equities. Meta object consists of general information about the requested symbol. Time series is the array of objects ordered by time descending with Open, High, Low, Close prices + Volume."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        outputsize: Default `30`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/time_series"
    querystring = {'symbol': symbol, 'interval': interval, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def real_time_price(symbol: str, format: str='json', outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is a lightweight method that allows retrieving only the real-time price of the selected instrument."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        format: Value can be `CSV` or `JSON`
Default `JSON`
        outputsize: Default `30`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/price"
    querystring = {'symbol': symbol, }
    if format:
        querystring['format'] = format
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quote(symbol: str, interval: str, outputsize: int=30, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Quote endpoint is an efficient method to retrieve the latest quote of the selected instrument."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        outputsize: Default `30`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/quote"
    querystring = {'symbol': symbol, 'interval': interval, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ln(symbol: str, interval: str, outputsize: int=30, format: str='json', series_type: str='close', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Natural Logarithm to the base of constant e(LN) transforms all data points with natural logarithm."
    symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/ln"
    querystring = {'symbol': symbol, 'interval': interval, }
    if outputsize:
        querystring['outputsize'] = outputsize
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ppo(interval: str, symbol: str, ma_type: str='SMA', format: str='json', series_type: str='close', slow_period: int=26, fast_period: int=12, outputsize: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Percentage Price Oscillator(PPO) shows the relationship between two Moving Averages(MA) as a percentage."
    interval: Interval between two consecutive points in time series
Supports: `1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `4h`, `1day`, `1week`, `1month`
        symbol: Instrument symbol, can be any equity, index, ETF, forex or cryptocurrency
E.g. `AAPL`, `EUR/USD`, `ETH/BTC`, ...
        ma_type: Supports: `SMA`, `EMA`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `KAMA`, `MAMA`, `T3MA`
        format: Value can be `CSV` or `JSON`
Default `JSON`
        series_type: Supports: `open`, `high`, `low`, `close`
        outputsize: Default `30` when no date parameters are set, otherwise set to maximum
        
    """
    url = f"https://twelve-data1.p.rapidapi.com/ppo"
    querystring = {'interval': interval, 'symbol': symbol, }
    if ma_type:
        querystring['ma_type'] = ma_type
    if format:
        querystring['format'] = format
    if series_type:
        querystring['series_type'] = series_type
    if slow_period:
        querystring['slow_period'] = slow_period
    if fast_period:
        querystring['fast_period'] = fast_period
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

