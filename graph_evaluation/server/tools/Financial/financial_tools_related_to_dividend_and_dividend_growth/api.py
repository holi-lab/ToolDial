import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calcyield(symb: str, period: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This tool calculates the average dividend yield over a given period. The period is in years, but can be any positive real number. For example period=1.5 represents one and a half year. symb is the ticker symbol of the security. The output represents the yield. To obtain the yield in percent, multiply by 100."
    symb: symb is the ticker symbol of the security.
        period: The period is in years, but can be any positive real number. For example period=1.5 represents one and a half year.
        
    """
    url = f"https://financial-tools-related-to-dividend-and-dividend-growth.p.rapidapi.com/"
    querystring = {'symb': symb, }
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-tools-related-to-dividend-and-dividend-growth.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calctotreturn(period: int, symb: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Total return is the amount of value an investor earns from a security over a specific period, when all distributions are reinvested.  The period is in years, but can be any positive real number. For example period=1.5 represents one and a half year. symb is the ticker symbol of the security. The output represents the total return. To obtain the total return in percent, multiply by 100."
    period: The period is in years, but can be any positive real number. For example period=1.5 represents one and a half year.
        symb: symb is the ticker symbol of the security. 
        
    """
    url = f"https://financial-tools-related-to-dividend-and-dividend-growth.p.rapidapi.com/"
    querystring = {'period': period, 'symb': symb, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-tools-related-to-dividend-and-dividend-growth.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calccagr(period: int, mysymb: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This growth rate is the Compound Annual Growth Rate (CAGR) in cash dividends per share of common stock over a given number of years (period). Dividend growth is a good indicator of the financial health of a company. mySymb is the ticker symbol of the security."
    period: Number of years backwards for calculating the CAGR. Must be a positive integer.
        mysymb: mySymb is the ticker symbol of the security.
        
    """
    url = f"https://financial-tools-related-to-dividend-and-dividend-growth.p.rapidapi.com/"
    querystring = {'period': period, 'mySymb': mysymb, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-tools-related-to-dividend-and-dividend-growth.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calcpayoutratio(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The dividend payout ratio is the ratio of the total amount of dividends paid out to shareholders relative to the net income of the company. It is the percentage of earnings paid to shareholders in dividends.
		Here, the trailing 12 months dividend paid is divided by the trailing 12 months GAAP net income.
		The input is the ticker symbol of the security. The output is the payout ratio. A value greater than 1 means that the dividend paid in the last 12 months exceeded the company net income."
    symbol: Ticker name of the security
        
    """
    url = f"https://financial-tools-related-to-dividend-and-dividend-growth.p.rapidapi.com/"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financial-tools-related-to-dividend-and-dividend-growth.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

