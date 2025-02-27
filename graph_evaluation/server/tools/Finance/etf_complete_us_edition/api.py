import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_etf_detail_info(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide a US listed ETF symbol as parameter and get a JSON response with general information about the ETF, like Title, descripion, Index, Issuer, Exchange, Currency, Class, Sector, Region, Inception Date and Expense Ratio.
		Get trading information about PE Ratio, current price, lows, highs, assets, NAV, outstanding shares and volume.
		Get performance information about the returns in months, years and YTD. 
		Get information about the exposure of the ETF by Region, Contry, Sector,  Market Cap and Asset Allocation.
		Get Technical information about volatility, indicators like RSI and MACD, support and resistance and more.
		Get the general dividend information like payout ratio, yield, annual dividend, frequency and growth,  and payout history of the ETF.
		Get the top 200 holdings within the ETF. The symbol, name, weighting and amount of shares is returned for each stock in the ETF"
    
    """
    url = f"https://etf-complete-us-edition.p.rapidapi.com/ETF/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etf-complete-us-edition.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_etf_list_with_exposure_to_provided_stock_symbol(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provide a stock symbol and get a list of ETFs with exposure to the provided stock.
		For the top 20 ETFs with exposure to the stock the Ticker, ETF name, Expense Ratio and Weighting within the ETF is shown."
    
    """
    url = f"https://etf-complete-us-edition.p.rapidapi.com/STOCK/{symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "etf-complete-us-edition.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

