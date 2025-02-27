import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def type_parent_get_investment_flows(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get investment flows as displayed in the Parent tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/parent/get-investment-flows"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_parent_get_summary(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get summary information as displayed in the Parent tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/parent/get-summary"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_get_strategy(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get strategy"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/get-strategy"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_people_get_proxy_voting_shareholder(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get proxy voting shareholder as displayed in the People tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/people/get-proxy-voting-shareholder"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_portfolio_get_holdings(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get holdings as displayed in the Portfolio tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/portfolio/get-holdings"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_portfolio_get_fixed_income_exposure_analysis(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fixed income exposure analysis as displayed in the Portfolio tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/portfolio/get-fixed-income-exposure-analysis"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_portfolio_get_maturity_schedule(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get maturity schedule as displayed in the Portfolio tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/portfolio/get-maturity-schedule"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_portfolio_get_historical_fixed_income_style(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get historical fixed income style as displayed in the Portfolio tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/portfolio/get-historical-fixed-income-style"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_portfolio_get_current_fixed_income_style(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current fixed income style as displayed in the Portfolio tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/portfolio/get-current-fixed-income-style"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_portfolio_get_asset_allocation(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get asset allocation as displayed in the Portfolio tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/portfolio/get-asset-allocation"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_price_get_other_fees(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get other fees as displayed in the Price tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/price/get-other-fees"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_price_get_taxes(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get taxes as displayed in the Price tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/price/get-taxes"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_price_get_fee_level(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fee level as displayed in the Price tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/price/get-fee-level"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_price_get_cost_illustration(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get cost illustration as displayed in the Price tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/price/get-cost-illustration"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_risk_get_risk_volatility_measures(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get risk volatility measures in the Risk tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/risk/get-risk-volatility-measures"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_risk_get_market_volatility_measures(securityid: str, type: str, year: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market volatility measures in the Risk tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/risk/get-market-volatility-measures"
    querystring = {'securityId': securityid, }
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_sustainability_get_esg_risk(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ESG risk in the Sustainability tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/sustainability/get-esg-risk"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_sustainability_get_product_involvement(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get product involvement in the Sustainability tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/sustainability/get-product-involvement"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_performance_get_trailing_returns(securityid: str, type: str, duration: str='daily', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trailing returns in the Performance tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        duration: One of the following : daily|monthly|quarterly
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/performance/get-trailing-returns"
    querystring = {'securityId': securityid, }
    if duration:
        querystring['duration'] = duration
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_performance_get_returns(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get returns in the Performance tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/performance/get-returns"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_analysis_get_archived(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Morningstar's analysis archived related to an ETF or FUND"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/analysis/get-archived"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_get_morningstar_analysis(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Morningstar's analysis related to an ETF or FUND"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/get-morningstar-analysis"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_get_disclosure_flag(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get disclosure flag related to an ETF or FUND"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/get-disclosure-flag"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_key_stats_get_operating_efficiency(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get key stats operating efficiency in the Valuation tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/key-stats/get-operating-efficiency"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_key_stats_get_financial_health(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get key stats financial health in the Valuation tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/key-stats/get-financial-health"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_news(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news in the News tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-news"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_trading_information(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Trading Information in the Quote tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-trading-information"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_parent_get_highest_rated_investments(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get highest rated investments as displayed in the Parent tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/parent/get-highest-rated-investments"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_parent_get_lowest_rated_investments(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get lowest rated investments as displayed in the Parent tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/parent/get-lowest-rated-investments"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_parent_get_morningstar_rating(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Morningstar's rating as displayed in the Parent tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/parent/get-morningstar-rating"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_parent_get_recent_investment_rating_change(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get recent investment rating change as displayed in the Parent tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/parent/get-recent-investment-rating-change"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_parent_get_top_medalist_rated_investments(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top medalist rated investments as displayed in the Parent tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/parent/get-top-medalist-rated-investments"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_parent_get_medalist_rating(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get medalist rating as displayed in the Parent tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/parent/get-medalist-rating"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_people_get_proxy_voting_management(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get proxy voting management as displayed in the People tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/people/get-proxy-voting-management"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_people_get_summary(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get summary information as displayed in the People tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/people/get-summary"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_portfolio_get_exposure(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get exposure as displayed in the Portfolio tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/portfolio/get-exposure"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_portfolio_get_coupon_range(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get coupon range as displayed in the Portfolio tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/portfolio/get-coupon-range"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_portfolio_get_credit_quality(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get credit quality as displayed in the Portfolio tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/portfolio/get-credit-quality"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_risk_get_risk_return_analysis(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get risk return analysis in the Risk tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/risk/get-risk-return-analysis"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_risk_get_risk_return_summary(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get summary of risk return in the Risk tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/risk/get-risk-return-summary"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_sustainability_get_carbon_metrics(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get carbon metrics in the Sustainability tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/sustainability/get-carbon-metrics"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_performance_get_latest_distributions(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest distributions in the Performance tab"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/performance/get-latest-distributions"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_performance_get_annual_distributions(securityid: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get annual distributions in the Performance tab"
    securityid: The value of securityId field returned in …/market/v2/get-returns
        type: One of the following values : etf|fund
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/performance/get-annual-distributions"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_analysis_get_comparables(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Morningstar's analysis comparables related to an ETF or FUND"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/analysis/get-comparables"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_get_mini_chart_realtime_data(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data to draw mini chat"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/get-mini-chart-realtime-data"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_get_realtime_data(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get realtime data related to an ETF or FUND"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/get-realtime-data"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def type_get_quote(type: str, securityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get quote related to an ETF or FUND"
    type: One of the following values : etf|fund
        securityid: The value of securityId field returned in …/market/v2/get-returns
        
    """
    url = f"https://ms-finance.p.rapidapi.com/{type}/get-quote"
    querystring = {'securityId': securityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_get_details(is_id: str, sourceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news details"
    id: Value of id returned in …/news/list endpoint
        sourceid: Value of sourceId returned in …/news/list endpoint
        
    """
    url = f"https://ms-finance.p.rapidapi.com/news/get-details"
    querystring = {'id': is_id, 'sourceId': sourceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/news/list"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles_get_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get articles details"
    id: Value of id returned in …/articles/list endpoint
        
    """
    url = f"https://ms-finance.p.rapidapi.com/articles/get-details"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles_list(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest articles"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/articles/list"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_v2_get_quotes_deprecated(performanceids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query one or multiple quotes at once
		* This endpoint is deprecated, use .../stock/v2/get-realtime-data endpoint instead"
    performanceids: The value of performanceId returned in …/market/v2/get-movers , …/market/v2/auto-complete, etc… endpoints. Separated by comma to query multiple entities.
        
    """
    url = f"https://ms-finance.p.rapidapi.com/market/v2/get-quotes"
    querystring = {'performanceIds': performanceids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_movers_deprecated(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top 10 gainers, losers, actives of specific market
		* * This endpoint is deprecated, use .../market/v2/get-movers endpoint instead"
    performanceid: Get value of PerformanceId field from /market/auto-complete and /market/get-summary APIs
        
    """
    url = f"https://ms-finance.p.rapidapi.com/market/get-movers"
    querystring = {'PerformanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_summary(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get live summary information at the request time"
    
    """
    url = f"https://ms-finance.p.rapidapi.com/market/get-summary"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_auto_complete(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto complete suggestion by term or phrase"
    query: The query value to get auto complete suggestions
        
    """
    url = f"https://ms-finance.p.rapidapi.com/market/auto-complete"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_v2_get_time_series(performanceids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query one or more entities data at once in time series manner"
    performanceids: The value of performanceId returned in .../market/v2/get-movers , .../market/v2/auto-complete, etc... endpoints. Separated by comma to query multiple entities.
        
    """
    url = f"https://ms-finance.p.rapidapi.com/market/v2/get-time-series"
    querystring = {'performanceIds': performanceids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_v2_auto_complete(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto suggestion by word or phase"
    q: Any thing you are familiar with, stock, index, organization, etc...
        
    """
    url = f"https://ms-finance.p.rapidapi.com/market/v2/auto-complete"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_videos(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest videos in the market"
    
    """
    url = f"https://ms-finance.p.rapidapi.com/market/get-videos"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_global_indices(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information of global indices"
    
    """
    url = f"https://ms-finance.p.rapidapi.com/market/get-global-indices"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_commentaries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest commentaries in the market"
    
    """
    url = f"https://ms-finance.p.rapidapi.com/market/get-commentaries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_articles(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest articles in the market"
    
    """
    url = f"https://ms-finance.p.rapidapi.com/market/get-articles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_v2_get_returns(performanceids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market return by years"
    performanceids: The value of performanceId returned in .../market/v2/get-movers , .../market/v2/auto-complete, etc... endpoints. Separated by comma to query multiple entities.
        
    """
    url = f"https://ms-finance.p.rapidapi.com/market/v2/get-returns"
    querystring = {'performanceIds': performanceids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_v2_get_realtime_data(performanceids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query one or more entities data at once"
    performanceids: The value of performanceId returned in .../market/v2/get-movers , .../market/v2/auto-complete, etc... endpoints. Separated by comma to query multiple entities.
        
    """
    url = f"https://ms-finance.p.rapidapi.com/market/v2/get-realtime-data"
    querystring = {'performanceIds': performanceids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_v2_get_movers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get movers, such as : actives, gainers, losers,  etc..."
    
    """
    url = f"https://ms-finance.p.rapidapi.com/market/v2/get-movers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_histories(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get prices from past to recent"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/get-histories"
    querystring = {'PerformanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v3_get_profile(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Company Profile section in Quote tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v3/get-profile"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_quote_deprecated(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock quote information"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/get-quote"
    querystring = {'PerformanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_key_stats_get_growth_table(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get key stats growth in the Valuation tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/key-stats/get-growth-table"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_dividends(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Dividends tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-dividends"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_competitors(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Competitors section in Quote tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-competitors"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_analysis_data(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Analysis section in Quote tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-analysis-data"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_trailing_total_returns(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Trailing Returns tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-trailing-total-returns"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_analysis_report(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get analysis report by experts"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-analysis-report"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_mini_chart_realtime_data(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to draw chart in Quote tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-mini-chart-realtime-data"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_security_info(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Quote section in Quote tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-security-info"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_realtime_data(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of stock, market,index, etc..."
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-realtime-data"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_key_stats_get_cash_flow(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get key stats cash flow in the Valuation tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/key-stats/get-cash-flow"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_risk_rating_assessment(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get risk rating assessment in the Sustainability tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-risk-rating-assessment"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_key_stats_get_overview(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get key stats overview in the Valuation tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/key-stats/get-overview"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_risk_rating_breakdown(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get risk rating breakdown in the Sustainability tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-risk-rating-breakdown"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_detail(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of stock, market,index, etc..."
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/get-detail"
    querystring = {'PerformanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_instruments(instrumentids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Competitors section in Quote tab"
    instrumentids: Value of instrumentId field from …/stock/v2/get-competitors endpoint
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-instruments"
    querystring = {'instrumentIds': instrumentids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_operating_performance(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Operating Performance tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-operating-performance"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_short_interest(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Short Interest section in Quote tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-short-interest"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_valuation(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Valuation tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-valuation"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_ownership(performanceid: str, asset: str='mutualfund', ownership: str='ConcentratedOwners', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Ownership tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        asset: One of the following : mutualfund | institution
        ownership: One of the following : OwnershipData | ConcentratedOwners | Sellers | Buyers
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-ownership"
    querystring = {'performanceId': performanceid, }
    if asset:
        querystring['asset'] = asset
    if ownership:
        querystring['ownership'] = ownership
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_profile(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Company Profile section in Quote tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-profile"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_mini_chart_quote(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to draw chart in Quote tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-mini-chart-quote"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_executive(performanceid: str, executive: str='keyExecutives', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Executive tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        executive: One of the following : keyExecutives | boardOfDirectors | committees | transactionHistory | transactionChart
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-executive"
    querystring = {'performanceId': performanceid, }
    if executive:
        querystring['executive'] = executive
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_splits(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Dividends tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-splits"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_financial_details(performanceid: str, datatype: str='A', reporttype: str='A', type: str='incomeStatement', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Expand Details View section found in Financials tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        datatype: One of the following : A - Annual | Q - Quarterly
        reporttype: One of the following : R - Restated | A - As originally reported
        type: One of the following : balanceSheet|cashFlow|incomeStatement
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-financial-details"
    querystring = {'performanceId': performanceid, }
    if datatype:
        querystring['dataType'] = datatype
    if reporttype:
        querystring['reportType'] = reporttype
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_financials(performanceid: str, reporttype: str='A', interval: str='annual', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Financials tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        reporttype: One of the following : R - Restated | A - As originally reported
        interval: One of the following : quarterly | annual
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-financials"
    querystring = {'performanceId': performanceid, }
    if reporttype:
        querystring['reportType'] = reporttype
    if interval:
        querystring['interval'] = interval
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_price_fair_value(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Price vs Fair Value tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-price-fair-value"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_v2_get_key_stats(performanceid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Mapped to Key Ratios section in Quote tab"
    performanceid: Value of performanceId field from …/auto-complete or /get-summary or …/get-movers endpoints
        
    """
    url = f"https://ms-finance.p.rapidapi.com/stock/v2/get-key-stats"
    querystring = {'performanceId': performanceid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ms-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

