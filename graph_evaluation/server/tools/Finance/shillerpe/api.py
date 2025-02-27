import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_30_year_interest_rate(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/interestrates/30-year-real-interest-rate/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_20_year_interest_rate(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/interestrates/20-year-real-interest-rate/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_10_year_interest_rate(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/interestrates/10-year-real-interest-rate/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_7_year_interest_rate(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/interestrates/7-year-real-interest-rate/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_5_year_interest_rate(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/interestrates/5-year-real-interest-rate/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_30_year_treasury_rate(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/treasuryrates/30-year-treasury-rate/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_20_year_treasury_rate(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/treasuryrates/20-year-treasury-rate/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_10_year_treasury_rate(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/treasuryrates/10-year-treasury-rate/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_7_year_treasury_rate(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/treasuryrates/7-year-treasury-rate/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_5_year_treasury_rate(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/treasuryrates/5-year-treasury-rate/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_3_year_treasury_rate(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/treasuryrates/3-year-treasury-rate/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_2_year_treasury_rate(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/treasuryrates/2-year-treasury-rate/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_year_treasury_rate(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/treasuryrates/1-year-treasury-rate/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_6_month_treasury_rate(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/treasuryrates/6-month-treasury-rate/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_3_month_treasury_rate(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/treasuryrates/3-month-treasury-rate/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shiller_pe(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/shiller-pe/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pe_ratio(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/pe-ratio/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_prices(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/historical-prices/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earnings(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/earnings/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earnings_yield(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/earnings-yield/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dividend(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/dividend/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dividend_yield(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/dividend-yield/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sales(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/sales/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sales_growth(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/sales-growth/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def real_sales(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/real-sales/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def real_sales_growth(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/real-sales-growth/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def real_earnings_growth(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/real-earnings-growth/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def price_to_sales(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/price-to-sales/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def price_to_book(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/price-to-book/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earnings_growth(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/earnings-growth/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def d_ividend_growth(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/dividend-growth/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def book_value_per_share(timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "S&P500 price to book value ratio. Current price to book ratio is estimated based on current market price and S&P 500 book value as of December, 2021 the latest reported by S&P."
    timeframe: Specify the desired timeframe
        
    """
    url = f"https://shillerpe.p.rapidapi.com/api/v1/sp500stats/book-value/{timeframe}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shillerpe.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

