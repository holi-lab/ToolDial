import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def m2_money_stock_adjusted(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get monthly seasonally adjusted M2 monetary aggregate from the Federal Reserve.  M2 is the money supply that includes all elements of M1 as well as "near money."   Near money refers to savings deposits, money market securities, mutual funds, and other time deposits."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/M2-sa-monthly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m1_money_stock_adjusted(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get monthly seasonally adjusted M1 monetary aggregate from the Federal Reserve.  M1 is the money supply that is composed of physical currency and coin, demand deposits, travelers' checks, other checkable deposits, and negotiable order of withdrawal (NOW) accounts."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/M1-sa-monthly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m1_money_stock_not_adjusted(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly not seasonally adjusted M1 monetary aggregate from the Federal Reserve.  M1 is the money supply that is composed of physical currency and coin, demand deposits, travelers' checks, other checkable deposits, and negotiable order of withdrawal (NOW) accounts."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/M1-nsa-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m2_money_stock_not_adjusted(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get weekly not seasonally adjusted M2 monetary aggregate from the Federal Reserve.  M2 is the money supply that includes all elements of M1 as well as "near money."   Near money refers to savings deposits, money market securities, mutual funds, and other time deposits."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/M2-nsa-weekly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mortgage_rate_30_year_fixed(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get monthly 30-Year fixed mortgage rates."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/mortgage-rate-30Y-fixed-monthly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nonfarm_payrolls_adjusted(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get monthly seasonally adjusted nonfarm payrolls."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/nonfarm-payrolls-total-us-sa-monthly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nonfarm_payrolls_not_adjusted(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get monthly not seasonally adjusted nonfarm payrolls."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/nonfarm-payrolls-total-us-nsa-monthly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unemployment_rate_adjusted(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get monthly seasonally adjusted unemployment rate."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/unemployment-rate-us-sa-monthly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def unemployment_rate_not_adjusted(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get monthly not seasonally adjusted unemployment rate."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/unemployment-rate-us-nsa-monthly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consumer_price_index(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get monthly consumer price index referenced to year 2010."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/cpi-us-monthly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_10_year_bond_yield(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get monthly 10-Year government bond yields."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/government-bond-10-year-us-monthly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def non_manufacturing_pmi(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get monthly non-manufacturing Purchasing Managers' Index."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/non-manufacturing-pmi-us-monthly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def manufacturing_pmi(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get monthly manufacturing Purchasing Managers' Index."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/manufacturing-pmi-us-monthly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def consumer_sentiment_index(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get monthly Consumer Sentiment Index."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/UofM-consumer-sentiment-index-us-monthly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def inflation(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get monthly inflation rates."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/inflation-us-monthly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prime_loan_rates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily bank prime loan interest rates."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/prime-rate-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gdp_growth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get quarterly percent change of Gross Domestic Product annualized."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/gdp-pct-change-us-quarterly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fed_funds_overnight_rate(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily federal funds overnight rate at which commercial banks borrow and lend their excess reserves to each other overnight."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/ffr-daily"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fed_policy_rates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get daily interest rates paid by the Federal Reserve System on reserves."
    
    """
    url = f"https://u-s-economic-indicators.p.rapidapi.com/api/v1/resources/fed-policy-rates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "u-s-economic-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

