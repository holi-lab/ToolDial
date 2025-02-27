import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def indices(exchanges: str, enddate: str, timehorizon: str, startdate: str, lowcap: bool, limit: str='1000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get indices data powered by Token Metrics"
    
    """
    url = f"https://crypto-market-data-from-token-metrics.p.rapidapi.com/indices"
    querystring = {'exchanges': exchanges, 'endDate': enddate, 'timeHorizon': timehorizon, 'startDate': startdate, 'lowCap': lowcap, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-market-data-from-token-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def correlation(tokens: str, limit: str='1000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get correlation for tokens powered by Token Metrics"
    
    """
    url = f"https://crypto-market-data-from-token-metrics.p.rapidapi.com/correlation"
    querystring = {'tokens': tokens, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-market-data-from-token-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scenario_analysis(tokens: str, limit: str='1000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get scenario analysis for tokens powered by Token Metrics"
    
    """
    url = f"https://crypto-market-data-from-token-metrics.p.rapidapi.com/scenario-analysis"
    querystring = {'tokens': tokens, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-market-data-from-token-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quantmetrics_tier_2(tokens: str, date: str, limit: str='1000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get quant metrics for tokens powered by Token Metrics"
    
    """
    url = f"https://crypto-market-data-from-token-metrics.p.rapidapi.com/quantmetrics-tier-2"
    querystring = {'tokens': tokens, 'date': date, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-market-data-from-token-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quantmetrics_tier_1(date: str, tokens: str, limit: str='1000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get quant metrics for tokens powered by Token Metrics"
    
    """
    url = f"https://crypto-market-data-from-token-metrics.p.rapidapi.com/quantmetrics-tier-1"
    querystring = {'date': date, 'tokens': tokens, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-market-data-from-token-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sentiments(enddate: str, startdate: str, tokens: str, limit: str='1000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market sentiment for tokens powered by Token Metrics"
    
    """
    url = f"https://crypto-market-data-from-token-metrics.p.rapidapi.com/sentiments"
    querystring = {'endDate': enddate, 'startDate': startdate, 'tokens': tokens, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-market-data-from-token-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def price_prediction(date: str, tokens: str, limit: str='1000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get price prediction for tokens powered by Token Metrics"
    
    """
    url = f"https://crypto-market-data-from-token-metrics.p.rapidapi.com/price-prediction"
    querystring = {'date': date, 'tokens': tokens, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-market-data-from-token-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def price(tokens: str, startdate: str, enddate: str, page: str='1', limit: str='1000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market price for tokens"
    
    """
    url = f"https://crypto-market-data-from-token-metrics.p.rapidapi.com/price"
    querystring = {'tokens': tokens, 'startDate': startdate, 'endDate': enddate, }
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-market-data-from-token-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def resistance_support(enddate: str, tokens: str, startdate: str, limit: str='1000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get automated Resistance and Support for tokens powered  by Token Metrics"
    
    """
    url = f"https://crypto-market-data-from-token-metrics.p.rapidapi.com/resistance-support"
    querystring = {'endDate': enddate, 'tokens': tokens, 'startDate': startdate, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-market-data-from-token-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trading_indicator(tokens: str, limit: str='1000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access Trading Indicator powered by Token Metrics"
    
    """
    url = f"https://crypto-market-data-from-token-metrics.p.rapidapi.com/trading-indicator"
    querystring = {'tokens': tokens, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-market-data-from-token-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_indicator(startdate: str, enddate: str, limit: str='1000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access Market Indicator powered by Token Metrics"
    
    """
    url = f"https://crypto-market-data-from-token-metrics.p.rapidapi.com/market-indicator"
    querystring = {'startDate': startdate, 'endDate': enddate, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-market-data-from-token-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def investor_grades(startdate: str, enddate: str, tokens: str, limit: str='1000', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access Investor Grades for tokens"
    
    """
    url = f"https://crypto-market-data-from-token-metrics.p.rapidapi.com/investor-grades"
    querystring = {'startDate': startdate, 'endDate': enddate, 'tokens': tokens, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-market-data-from-token-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trader_grades(enddate: str, startdate: str, limit: str, tokens: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access Trader Grades for tokens"
    
    """
    url = f"https://crypto-market-data-from-token-metrics.p.rapidapi.com/trader-grades"
    querystring = {'endDate': enddate, 'startDate': startdate, 'limit': limit, 'tokens': tokens, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-market-data-from-token-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tokens(token_symbols: str='bds, bds, btc', token_names: str='Hivemapper, Sei_Network, Layer_Zero, Lyra Huobi', token_ids: str='3375, 3306', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get tokens"
    
    """
    url = f"https://crypto-market-data-from-token-metrics.p.rapidapi.com/tokens"
    querystring = {}
    if token_symbols:
        querystring['token_symbols'] = token_symbols
    if token_names:
        querystring['token_names'] = token_names
    if token_ids:
        querystring['token_ids'] = token_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "crypto-market-data-from-token-metrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

