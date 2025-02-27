import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stats_mse_and_mae_linear_regression_and_correlation_for_funding_rate_t_0_vs_future_funding_rate_t_x_days(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Correlation and linear regression analysis of the funding rate at any particular moment in time (day 0) with the future funding rate (day x, where x is larger than 0). This data shows how long a measured funding rate has influence on the funding rate before the correlation drops of to zero and it is random once again. For regression, mean squared error (MSE) and mean absolute error (MAE) are used. The starting point of the MAE is the slope and intercept obtained from the MSE. MAE does not necessarily find an absolute minimum, but can find a local minimum. By starting with the MSE as input for the MSE, the likelihood however greatly increases to find the absolute minimum error.
		
		**Note**
		
		So far I observed that the MSE slope is on average higher than the MAE slope for the same day. Since MSE squares the error, big deviations from expected funding rate dominate the error term and thus the resulting slope and intercept value. Apparently there  have been more 'outliers' to the upside, since the MSE slope is higher then the MAE slope."
    
    """
    url = f"https://ftx-tracker1.p.rapidapi.com/stats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ftx-tracker1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_ftx_borrow_rates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the borrow rates from FTX API. The borrow rates is what you pay to hold a net short balance on an asset."
    
    """
    url = f"https://ftx-tracker1.p.rapidapi.com/template"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ftx-tracker1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_ftx_trading_pairs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Direct request to the ftx api that returns all the listed pairs. Basic options are spots (usd/usdt), perp (perpetual future), normal future (f.e. 1231; future expiring december 31,  or each quartile last friday), move (contract for difference daily), Bull/Bear (3x) and more exotic options such as betting on bolsonaro next presidency."
    
    """
    url = f"https://ftx-tracker1.p.rapidapi.com/quotelist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ftx-tracker1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_my_portfolio_historic_performance(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns your portfolio value (timestamp,  1 FTX account per user for now). If you add your FTX API  in /adduser the bot will start tracking your portfolio value here."
    
    """
    url = f"https://ftx-tracker1.p.rapidapi.com/myBalances"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ftx-tracker1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_ticker_future_perpetual_pair_returns(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the return including leverage of the optimal long short future perpetual or vice versa of all trading pairs."
    
    """
    url = f"https://ftx-tracker1.p.rapidapi.com/futperp"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ftx-tracker1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_ticker_carry_returns(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the annual non-compounded return of carry trades, e.g. future-spot pairs. A long carry trade would sell the future (which is at a premium to the spot due to the opportunity value of getting selling leverage) and buy the spot."
    
    """
    url = f"https://ftx-tracker1.p.rapidapi.com/carry"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ftx-tracker1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_info_per_ticker_used_for_return_calculations(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for an individual ticker with: /breakdown/ticker f.e. /breakdown/BTC 
		Output is shows API information found from FTX regarding the futures, spot data and calculation of APR/APY for the Carry and FutPerp returns."
    
    """
    url = f"https://ftx-tracker1.p.rapidapi.com/breakdown/{ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ftx-tracker1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def check_last_update_for_fundrate_database(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The bot runs every 10 min. If the bot is up to date, the funding rate time in UTC should be displayed."
    
    """
    url = f"https://ftx-tracker1.p.rapidapi.com/database_stats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ftx-tracker1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

