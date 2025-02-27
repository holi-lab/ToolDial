import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def symbols_get_metrics(symbols: str, fields: str='quant_rating,authors_rating_pro,sell_side_rating,marketcap,dividend_yield,value_category,growth_category,profitability_category,momentum_category,eps_revisions_category', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get metrics of specific symbols"
    symbols: Symbol to query for data.  Separating  by comma to query multiple symbols at once, ex : aapl,tsla
        fields: One of the following : altman&#95;z&#95;score|analysts&#95;down&#95;avg&#95;5y|analysts&#95;down&#95;percent&#95;avg&#95;5y|analysts&#95;up&#95;avg&#95;5y|analysts&#95;up&#95;percent&#95;avg&#95;5y|assets&#95;turnover|authors&#95;rating&#95;pro|beta24|capex&#95;change|capex&#95;change&#95;avg&#95;5y|capex&#95;to&#95;sales|cash&#95;from&#95;operations&#95;as&#95;reported|cf&#95;op&#95;change&#95;display|cf&#95;op&#95;change&#95;display&#95;avg&#95;5y|coefficient&#95;of&#95;variation&#95;90d|common&#95;equity&#95;10y|common&#95;equity&#95;3y|common&#95;equity&#95;5y|common&#95;equity&#95;yoy|degree&#95;of&#95;operating&#95;leverage&#95;ttm|diluted&#95;eps&#95;growth|diluted&#95;eps&#95;growth&#95;avg&#95;5y|dilutedEps10y|dilutedEps3y|dilutedEps5y|dilutedEpsGrowth|div&#95;grow&#95;rate10|div&#95;grow&#95;rate3|div&#95;grow&#95;rate5|div&#95;growth&#95;category|div&#95;pay&#95;date|div&#95;rate&#95;fwd|div&#95;rate&#95;ttm|div&#95;yield&#95;4y&#95;avg&#95;5y|div&#95;yield&#95;category&#95;avg&#95;5y|div&#95;yield&#95;fwd|div&#95;yield&#95;fwd&#95;avg&#95;5y|dividend&#95;growth|dividend&#95;lt&#95;fwd&#95;growth|dividend&#95;per&#95;share&#95;change&#95;dislpay|dividend&#95;per&#95;share&#95;change&#95;dislpay&#95;avg&#95;5y|dividend&#95;yield|dividend&#95;yield&#95;avg&#95;5y|dps&#95;yoy|dps&#95;yoy&#95;avg&#95;5y|earn&#95;yield&#95;gaap&#95;fy1&#95;avg&#95;5y|earnings&#95;yield&#95;avg&#95;5y|earningsGrowth|earningsGrowth10y|earningsGrowth3|earningsGrowth5y|ebit&#95;change&#95;display|ebit&#95;change&#95;display&#95;avg&#95;5y|ebit&#95;margin|ebitda&#95;10y|ebitda&#95;3y|ebitda&#95;5y|ebitda&#95;change&#95;display|ebitda&#95;change&#95;display&#95;avg&#95;5y|ebitda&#95;margin|ebitda&#95;yoy|ebitda&#95;yoy&#95;avg&#95;5y|ebitdaYoy|eps&#95;change&#95;display|eps&#95;change&#95;display&#95;avg&#95;5y|eps&#95;ltg|eps&#95;ltg&#95;avg&#95;5y|eps&#95;revisions&#95;category|ev&#95;12m&#95;sales&#95;ratio|ev&#95;ebitda|fcf&#95;per&#95;share&#95;change&#95;display|fcf&#95;per&#95;share&#95;change&#95;display&#95;avg&#95;5y|fcf&#95;yield&#95;avg&#95;5y|fcf&#95;yield&#95;fy1&#95;avg&#95;5y|gross&#95;loans&#95;10y|gross&#95;loans&#95;3y|gross&#95;loans&#95;5y|gross&#95;loans&#95;yoy|gross&#95;margin|growth&#95;category|impliedmarketcap|last&#95;div&#95;date|last&#95;price&#95;vs&#95;sma&#95;10d|last&#95;price&#95;vs&#95;sma&#95;200d|last&#95;price&#95;vs&#95;sma&#95;50d|levered&#95;fcf&#95;margin|levered&#95;free&#95;cash&#95;flow&#95;yoy|levered&#95;free&#95;cash&#95;flow&#95;yoy&#95;avg&#95;5y|leveredFreeCashFlow10y|leveredFreeCashFlow3y|leveredFreeCashFlow5y|leveredFreeCashFlowYoy|log&#95;of&#95;unadjusted&#95;stock&#95;price|marketcap|marketcap&#95;display|momentum&#95;category|net&#95;eps|net&#95;inc&#95;per&#95;employee|net&#95;income|net&#95;interest&#95;income&#95;10y|net&#95;interest&#95;income&#95;3y|net&#95;interest&#95;income&#95;5y|net&#95;interest&#95;income&#95;yoy|net&#95;margin|netIncome10y|netIncome3y|netIncome5y|netIncomeYoy|normalizedNetIncome10y|normalizedNetIncome3y|normalizedNetIncome5y|normalizedNetIncomeYoy|op&#95;cf&#95;yoy|op&#95;cf&#95;yoy&#95;avg&#95;5y|oper&#95;income&#95;fy1&#95;market&#95;cap&#95;avg&#95;5y|oper&#95;income&#95;market&#95;cap&#95;avg&#95;5y|operating&#95;income&#95;ebit&#95;yoy|operating&#95;income&#95;ebit&#95;yoy&#95;avg&#95;5y|operatingIncomeEbit10y|operatingIncomeEbit3y|operatingIncomeEbit5y|operatingIncomeEbitYoy|payout&#95;ratio|pb&#95;ratio|pe&#95;nongaap&#95;fy1|pe&#95;ratio|price&#95;cf&#95;ratio|price&#95;high&#95;52w|price&#95;low&#95;52w|profitability&#95;category|quant&#95;rating|return&#95;on&#95;avg&#95;tot&#95;assets|return&#95;on&#95;net&#95;tangible&#95;assets|return&#95;on&#95;total&#95;capital|revenue&#95;change&#95;display|revenue&#95;change&#95;display&#95;avg&#95;5y|revenue&#95;growth|revenue&#95;growth&#95;avg&#95;5y|revenue&#95;growth3|revenue&#95;growth5|revenueGrowth10|roe|roe&#95;change&#95;display|roe&#95;change&#95;display&#95;avg&#95;5y|roe&#95;yoy|roe&#95;yoy&#95;avg&#95;5y|rtn&#95;on&#95;common&#95;equity|sell&#95;side&#95;rating|shares|short&#95;interest&#95;percent&#95;of&#95;float|sma&#95;10d|sma&#95;200d|sma&#95;50d|tangible&#95;book&#95;per&#95;share|tangibleBookValue10y|tangibleBookValue3y|tangibleBookValue5y|tangibleBookValueYoy|tev|total&#95;cash|total&#95;debt|total&#95;revenue|totalAssets10y|totalAssets3y|totalAssets5y|totalAssetsYoy|value&#95;category|working&#95;cap&#95;change|working&#95;cap&#95;change&#95;avg&#95;5y|yld&#95;on&#95;cost&#95;1y&#95;avg&#95;5y|yld&#95;on&#95;cost&#95;3y&#95;avg&#95;5y|yld&#95;on&#95;cost&#95;5y&#95;avg&#95;5y . 
Separated by comma for multiple options. Ex : quant&#95;rating,authors&#95;rating&#95;pro,sell&#95;side&#95;rating,marketcap,dividend&#95;yield,etc...
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-metrics"
    querystring = {'symbols': symbols, }
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_earnings(ticker_ids: str, revisions_data_items: str='eps_normalized_actual,eps_normalized_consensus_mean,revenue_consensus_mean', estimates_data_items: str='eps_gaap_actual,eps_gaap_consensus_mean,eps_normalized_actual,eps_normalized_consensus_mean,revenue_actual,revenue_consensus_mean', relative_periods: str='-3,-2,-1,0,1,2,3', group_by_month: bool=None, return_window: int=None, period_type: str='quarterly', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information  in Earnings tab of specific symbol"
    ticker_ids: The value of id fields returned in .../symbols/get-meta-data endpoint.  Separating  by comma to query multiple tickers at once, ex : 1742,146
        revisions_data_items: One of the followings : eps&#95;normalized&#95;actual,eps&#95;normalized&#95;consensus&#95;mean,revenue&#95;consensus&#95;mean . Separated by comma for multiple options
        estimates_data_items: One of the followings : eps&#95;gaap&#95;actual,eps&#95;gaap&#95;consensus&#95;mean,eps&#95;normalized&#95;actual,eps&#95;normalized&#95;consensus&#95;mean,revenue&#95;actual,revenue&#95;consensus&#95;mean,eps&#95;primary . Separated by comma for multiple options
        relative_periods: Valid range  -23,...,-2,-1,0,1,2,..,23
        group_by_month: true|false
        period_type: One of the followings : quarterly|annual
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-earnings"
    querystring = {'ticker_ids': ticker_ids, }
    if revisions_data_items:
        querystring['revisions_data_items'] = revisions_data_items
    if estimates_data_items:
        querystring['estimates_data_items'] = estimates_data_items
    if relative_periods:
        querystring['relative_periods'] = relative_periods
    if group_by_month:
        querystring['group_by_month'] = group_by_month
    if return_window:
        querystring['return_window'] = return_window
    if period_type:
        querystring['period_type'] = period_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_momentum_deprecated(symbols: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get momentum of specific symbol
		* This endpoint is deprecated,  you need to use .../symbols/v2/get-momentum endpoint instead."
    symbols: Symbol to query for data.  Separating  by comma to query multiple symbols at once, ex : aapl,tsla
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-momentum"
    querystring = {'symbols': symbols, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_options(identifier: str, month: int, year: int, identifiertype: str='Symbol', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get optional prices"
    identifier: Symbol to query for data, only one is allowed at a time.
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-options"
    querystring = {'Identifier': identifier, 'Month': month, 'Year': year, }
    if identifiertype:
        querystring['IdentifierType'] = identifiertype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_option_expirations(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get option expirations to use with .../symbols/get-options endpoint"
    symbol: Symbol to query for data. 
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-option-expirations"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_sector_metrics(symbol: str, fields: str='gross_margin,ebit_margin,ebitda_margin,net_margin,levered_fcf_margin,rtn_on_common_equity,return_on_total_capital,return_on_avg_tot_assets,capex_to_sales,assets_turnover,cash_from_operations_as_reported,net_inc_per_employee', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Profitability, Growth, etc... metrics"
    symbol: Symbol to query for data, only one is allowed at a time.
        fields: One of the following, separated by comma for multiple options : altman&#95;z&#95;score|analysts&#95;down&#95;avg&#95;5y|analysts&#95;down&#95;percent&#95;avg&#95;5y|analysts&#95;up&#95;avg&#95;5y|analysts&#95;up&#95;percent&#95;avg&#95;5y|assets&#95;turnover|authors&#95;rating&#95;pro|beta24|capex&#95;change|capex&#95;change&#95;avg&#95;5y|capex&#95;to&#95;sales|cash&#95;from&#95;operations&#95;as&#95;reported|cf&#95;op&#95;change&#95;display|cf&#95;op&#95;change&#95;display&#95;avg&#95;5y|common&#95;equity&#95;10y|common&#95;equity&#95;3y|common&#95;equity&#95;5y|common&#95;equity&#95;yoy|diluted&#95;eps&#95;growth|diluted&#95;eps&#95;growth&#95;avg&#95;5y|dilutedEps10y|dilutedEps3y|dilutedEps5y|dilutedEpsGrowth|div&#95;grow&#95;rate3|div&#95;grow&#95;rate5|div&#95;pay&#95;date|div&#95;rate&#95;fwd|div&#95;rate&#95;ttm|div&#95;yield&#95;fwd|dividend&#95;growth|dividend&#95;per&#95;share&#95;change&#95;dislpay|dividend&#95;per&#95;share&#95;change&#95;dislpay&#95;avg&#95;5y|dividend&#95;yield|dps&#95;yoy|dps&#95;yoy&#95;avg&#95;5y|earningsGrowth|earningsGrowth10y|earningsGrowth3|earningsGrowth5y|ebit&#95;change&#95;display|ebit&#95;change&#95;display&#95;avg&#95;5y|ebit&#95;margin|ebitda&#95;10y|ebitda&#95;3y|ebitda&#95;5y|ebitda&#95;change&#95;display|ebitda&#95;change&#95;display&#95;avg&#95;5y|ebitda&#95;margin|ebitda&#95;yoy|ebitda&#95;yoy&#95;avg&#95;5y|ebitdaYoy|eps&#95;change&#95;display|eps&#95;change&#95;display&#95;avg&#95;5y|eps&#95;ltg|eps&#95;ltg&#95;avg&#95;5y|eps&#95;revisions&#95;category|ev&#95;12m&#95;sales&#95;ratio|ev&#95;ebitda|fcf&#95;per&#95;share&#95;change&#95;display|fcf&#95;per&#95;share&#95;change&#95;display&#95;avg&#95;5y|gross&#95;loans&#95;10y|gross&#95;loans&#95;3y|gross&#95;loans&#95;5y|gross&#95;loans&#95;yoy|gross&#95;margin|growth&#95;category|impliedmarketcap|last&#95;div&#95;date|last&#95;price&#95;vs&#95;sma&#95;10d|last&#95;price&#95;vs&#95;sma&#95;200d|last&#95;price&#95;vs&#95;sma&#95;50d|levered&#95;fcf&#95;margin|levered&#95;free&#95;cash&#95;flow&#95;yoy|levered&#95;free&#95;cash&#95;flow&#95;yoy&#95;avg&#95;5y|leveredFreeCashFlow10y|leveredFreeCashFlow3y|leveredFreeCashFlow5y|leveredFreeCashFlowYoy|marketcap|marketcap&#95;display|momentum&#95;category|net&#95;eps|net&#95;inc&#95;per&#95;employee|net&#95;income|net&#95;interest&#95;income&#95;10y|net&#95;interest&#95;income&#95;3y|net&#95;interest&#95;income&#95;5y|net&#95;interest&#95;income&#95;yoy|net&#95;margin|netIncome10y|netIncome3y|netIncome5y|netIncomeYoy|normalizedNetIncome10y|normalizedNetIncome3y|normalizedNetIncome5y|normalizedNetIncomeYoy|op&#95;cf&#95;yoy|op&#95;cf&#95;yoy&#95;avg&#95;5y|operating&#95;income&#95;ebit&#95;yoy|operating&#95;income&#95;ebit&#95;yoy&#95;avg&#95;5y|operatingIncomeEbit10y|operatingIncomeEbit3y|operatingIncomeEbit5y|operatingIncomeEbitYoy|payout&#95;ratio|pb&#95;ratio|pe&#95;nongaap&#95;fy1|pe&#95;ratio|price&#95;cf&#95;ratio|price&#95;high&#95;52w|price&#95;low&#95;52w|profitability&#95;category|quant&#95;rating|return&#95;on&#95;avg&#95;tot&#95;assets|return&#95;on&#95;total&#95;capital|revenue&#95;change&#95;display|revenue&#95;change&#95;display&#95;avg&#95;5y|revenue&#95;growth|revenue&#95;growth&#95;avg&#95;5y|revenue&#95;growth3|revenue&#95;growth5|revenueGrowth10|roe|roe&#95;change&#95;display|roe&#95;change&#95;display&#95;avg&#95;5y|roe&#95;yoy|roe&#95;yoy&#95;avg&#95;5y|rtn&#95;on&#95;common&#95;equity|sell&#95;side&#95;rating|shares|short&#95;interest&#95;percent&#95;of&#95;float|sma&#95;10d|sma&#95;200d|sma&#95;50d|tangible&#95;book&#95;per&#95;share|tangibleBookValue10y|tangibleBookValue3y|tangibleBookValue5y|tangibleBookValueYoy|tev|total&#95;cash|total&#95;debt|total&#95;revenue|totalAssets10y|totalAssets3y|totalAssets5y|totalAssetsYoy|value&#95;category|working&#95;cap&#95;change|working&#95;cap&#95;change&#95;avg&#95;5y
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-sector-metrics"
    querystring = {'symbol': symbol, }
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_metric_grades(symbol: str, algos: str='main_quant,dividends', fields: str='gross_margin,ebit_margin,ebitda_margin,net_margin,levered_fcf_margin,rtn_on_common_equity,return_on_total_capital,return_on_avg_tot_assets,capex_to_sales,assets_turnover,cash_from_operations_as_reported,net_inc_per_employee', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Profitability, Growth, etc... grade"
    symbol: Symbol to query for data, only one is allowed at a time.
        algos: One of the following, separated by comma for multiple options : main_quant,dividends
        fields: One of the following, separated by comma for multiple options : altman&#95;z&#95;score|analysts&#95;down&#95;avg&#95;5y|analysts&#95;down&#95;percent&#95;avg&#95;5y|analysts&#95;up&#95;avg&#95;5y|analysts&#95;up&#95;percent&#95;avg&#95;5y|assets&#95;turnover|authors&#95;rating&#95;pro|beta24|capex&#95;change|capex&#95;change&#95;avg&#95;5y|capex&#95;to&#95;sales|cash&#95;from&#95;operations&#95;as&#95;reported|cf&#95;op&#95;change&#95;display|cf&#95;op&#95;change&#95;display&#95;avg&#95;5y|common&#95;equity&#95;10y|common&#95;equity&#95;3y|common&#95;equity&#95;5y|common&#95;equity&#95;yoy|diluted&#95;eps&#95;growth|diluted&#95;eps&#95;growth&#95;avg&#95;5y|dilutedEps10y|dilutedEps3y|dilutedEps5y|dilutedEpsGrowth|div&#95;grow&#95;rate3|div&#95;grow&#95;rate5|div&#95;pay&#95;date|div&#95;rate&#95;fwd|div&#95;rate&#95;ttm|div&#95;yield&#95;fwd|dividend&#95;growth|dividend&#95;per&#95;share&#95;change&#95;dislpay|dividend&#95;per&#95;share&#95;change&#95;dislpay&#95;avg&#95;5y|dividend&#95;yield|dps&#95;yoy|dps&#95;yoy&#95;avg&#95;5y|earningsGrowth|earningsGrowth10y|earningsGrowth3|earningsGrowth5y|ebit&#95;change&#95;display|ebit&#95;change&#95;display&#95;avg&#95;5y|ebit&#95;margin|ebitda&#95;10y|ebitda&#95;3y|ebitda&#95;5y|ebitda&#95;change&#95;display|ebitda&#95;change&#95;display&#95;avg&#95;5y|ebitda&#95;margin|ebitda&#95;yoy|ebitda&#95;yoy&#95;avg&#95;5y|ebitdaYoy|eps&#95;change&#95;display|eps&#95;change&#95;display&#95;avg&#95;5y|eps&#95;ltg|eps&#95;ltg&#95;avg&#95;5y|eps&#95;revisions&#95;category|ev&#95;12m&#95;sales&#95;ratio|ev&#95;ebitda|fcf&#95;per&#95;share&#95;change&#95;display|fcf&#95;per&#95;share&#95;change&#95;display&#95;avg&#95;5y|gross&#95;loans&#95;10y|gross&#95;loans&#95;3y|gross&#95;loans&#95;5y|gross&#95;loans&#95;yoy|gross&#95;margin|growth&#95;category|impliedmarketcap|last&#95;div&#95;date|last&#95;price&#95;vs&#95;sma&#95;10d|last&#95;price&#95;vs&#95;sma&#95;200d|last&#95;price&#95;vs&#95;sma&#95;50d|levered&#95;fcf&#95;margin|levered&#95;free&#95;cash&#95;flow&#95;yoy|levered&#95;free&#95;cash&#95;flow&#95;yoy&#95;avg&#95;5y|leveredFreeCashFlow10y|leveredFreeCashFlow3y|leveredFreeCashFlow5y|leveredFreeCashFlowYoy|marketcap|marketcap&#95;display|momentum&#95;category|net&#95;eps|net&#95;inc&#95;per&#95;employee|net&#95;income|net&#95;interest&#95;income&#95;10y|net&#95;interest&#95;income&#95;3y|net&#95;interest&#95;income&#95;5y|net&#95;interest&#95;income&#95;yoy|net&#95;margin|netIncome10y|netIncome3y|netIncome5y|netIncomeYoy|normalizedNetIncome10y|normalizedNetIncome3y|normalizedNetIncome5y|normalizedNetIncomeYoy|op&#95;cf&#95;yoy|op&#95;cf&#95;yoy&#95;avg&#95;5y|operating&#95;income&#95;ebit&#95;yoy|operating&#95;income&#95;ebit&#95;yoy&#95;avg&#95;5y|operatingIncomeEbit10y|operatingIncomeEbit3y|operatingIncomeEbit5y|operatingIncomeEbitYoy|payout&#95;ratio|pb&#95;ratio|pe&#95;nongaap&#95;fy1|pe&#95;ratio|price&#95;cf&#95;ratio|price&#95;high&#95;52w|price&#95;low&#95;52w|profitability&#95;category|quant&#95;rating|return&#95;on&#95;avg&#95;tot&#95;assets|return&#95;on&#95;total&#95;capital|revenue&#95;change&#95;display|revenue&#95;change&#95;display&#95;avg&#95;5y|revenue&#95;growth|revenue&#95;growth&#95;avg&#95;5y|revenue&#95;growth3|revenue&#95;growth5|revenueGrowth10|roe|roe&#95;change&#95;display|roe&#95;change&#95;display&#95;avg&#95;5y|roe&#95;yoy|roe&#95;yoy&#95;avg&#95;5y|rtn&#95;on&#95;common&#95;equity|sell&#95;side&#95;rating|shares|short&#95;interest&#95;percent&#95;of&#95;float|sma&#95;10d|sma&#95;200d|sma&#95;50d|tangible&#95;book&#95;per&#95;share|tangibleBookValue10y|tangibleBookValue3y|tangibleBookValue5y|tangibleBookValueYoy|tev|total&#95;cash|total&#95;debt|total&#95;revenue|totalAssets10y|totalAssets3y|totalAssets5y|totalAssetsYoy|value&#95;category|working&#95;cap&#95;change|working&#95;cap&#95;change&#95;avg&#95;5y
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-metric-grades"
    querystring = {'symbol': symbol, }
    if algos:
        querystring['algos'] = algos
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_historical_prices(symbol: str, start: str, end: str, show_by: str='week', sort: str='as_of_date', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get historical prices"
    symbol: Symbol to query for data, only one is allowed at a time.
        start: The date to get historical prices from. The format is yyyy-MM-dd . Ex : 2022-02-01
        end: The date to get historical prices to. The format is yyyy-MM-dd . Ex : 2023-03-09
        show_by: One of the following : day|week|month
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-historical-prices"
    querystring = {'symbol': symbol, 'start': start, 'end': end, }
    if show_by:
        querystring['show_by'] = show_by
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_ratings_deprecated(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ratings data for specific symbol
		* This endpoint is replaced by .../symbols/get-factor-grades and .../symbols/get-quant-rating-histories"
    symbol: Symbol to query for data, only one is allowed at a time.
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-ratings"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_analyst_price_target(ticker_ids: str, return_window: int=1, group_by_month: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Wall Street analyst price target for specific symbol"
    ticker_ids: The value of id field returned in .../symbols/get-meta-data
        return_window: The return window
        group_by_month: Whether or not the data is grouped by month
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-analyst-price-target"
    querystring = {'ticker_ids': ticker_ids, }
    if return_window:
        querystring['return_window'] = return_window
    if group_by_month:
        querystring['group_by_month'] = group_by_month
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_splits(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get splits"
    symbol: Symbol to query for data, only one is allowed at a time.
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-splits"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_dividend_history(symbol: str, years: str='6', group_by: str='year', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get dividend history of specific symbol"
    symbol: Symbol to query for data, only one is allowed at a time.
        group_by: One of the following : year|month
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-dividend-history"
    querystring = {'symbol': symbol, }
    if years:
        querystring['years'] = years
    if group_by:
        querystring['group_by'] = group_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_profile(symbols: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get profile information of specific symbol"
    symbols: Symbol to query for data.  Separating  by comma to query multiple symbols at once, ex : aapl,tsla
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-profile"
    querystring = {'symbols': symbols, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_fundamentals(symbol: str, limit: str='4', period_type: str='annual', field: str='revenue_per_share', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fundamentals for specific symbol"
    symbol: Symbol to query for data, only one is allowed at a time.
        period_type: One of the following : quarterly|annual
        field: One of the following : revenues|other&#95;revenues&#95;summary&#95;subtotal|total&#95;revenue|cost&#95;revenue|gross&#95;profit|selling&#95;general&#95;admin&#95;expenses&#95;total|rd&#95;expenses|other&#95;operating&#95;exp&#95;total|operating&#95;income|interest&#95;expense&#95;total|interest&#95;and&#95;investment&#95;income|net&#95;interest&#95;exp&#95;standard|currency&#95;exchange&#95;gains&#95;loss|other&#95;non&#95;operating&#95;income|ebt&#95;incl&#95;unusual&#95;items|income&#95;tax&#95;expense|earnings&#95;from&#95;cont&#95;ops|net&#95;income&#95;to&#95;company|net&#95;income|ni&#95;to&#95;common&#95;incl&#95;extra&#95;items|ni&#95;to&#95;common&#95;excl&#95;extra&#95;items|revenue&#95;per&#95;share|eps|basic&#95;eps&#95;excl&#95;extra&#95;items|weighted&#95;average&#95;basic&#95;shares&#95;outstanding|diluted&#95;eps|diluted&#95;eps&#95;excl&#95;extra&#95;itmes|weighted&#95;average&#95;diluted&#95;shares&#95;outstanding|normalized&#95;basic&#95;eps|normalized&#95;diluted&#95;eps|div&#95;rate|payout&#95;ratio|ebitda|ebita|ebit&#95;op&#95;in|ebitdar|effective&#95;tax&#95;rate|normalized&#95;net&#95;income|interest&#95;on&#95;long&#95;term&#95;debt|r&#95;d&#95;exp|foreign&#95;sales
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-fundamentals"
    querystring = {'symbol': symbol, }
    if limit:
        querystring['limit'] = limit
    if period_type:
        querystring['period_type'] = period_type
    if field:
        querystring['field'] = field
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_meta_data(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get meta data of specific symbol"
    symbol: Symbol to query for data, only one is allowed at a time.
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-meta-data"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_top_holdings_deprecated(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get top holdings of specific symbol"
    symbol: Symbol to query for data.  Separating  by comma to query multiple symbols at once, ex : aapl,tsla
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-top-holdings"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_factor_grades_deprecated(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get factor grades for specific symbol"
    symbol: Symbol to query for data, only one is allowed at a time.
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-factor-grades"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_key_data_deprecated(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get key data of specific symbol"
    symbol: Symbol to query for data, only one is allowed at a time.
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-key-data"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_v2_get_chart(end: str, start: str, symbols: str, metrics: str='total_revenue', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint reproduces public data and features in Charting tab."
    end: Ending date to query for data, the format is yyyy-MM-dd
        start: Starting date to query for data, the format is yyyy-MM-dd
        symbols: Symbol to query for data. Separating by comma to query multiple symbols at once, ex : aapl,tsla
        metrics: One of the following : total&#95;revenue|ebitda&#95;yoy|net&#95;income|diluted&#95;eps&#95;growth|pe&#95;ratio|pb&#95;ratio|price&#95;cf&#95;ratio|ps&#95;ratio|price&#95;tang&#95;book|ev&#95;ebit|ev&#95;12m&#95;sales&#95;ratio|enterprise&#95;value|ev&#95;ebitda|market&#95;cap|gross&#95;margin|net&#95;margin|operating&#95;income&#95;ebit&#95;yoy|normalized&#95;net&#95;income|tangible&#95;book|total&#95;assets|levered&#95;free&#95;cash&#95;flow&#95;yoy|diluted&#95;weighted&#95;average&#95;shares&#95;outstanding|ebit&#95;margin|normalized&#95;net&#95;income&#95;margin|ebitda&#95;margin|levered&#95;fcf&#95;margin|return&#95;on&#95;equity|return&#95;on&#95;avg&#95;tot&#95;assets|return&#95;on&#95;total&#95;capital|assets&#95;turnover|net&#95;interest&#95;income|gross&#95;loans|total&#95;common&#95;equity|sga&#95;margin|ebt&#95;margin|net&#95;interest&#95;income&#95;per&#95;total&#95;revenue
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/v2/get-chart"
    querystring = {'end': end, 'start': start, 'symbols': symbols, }
    if metrics:
        querystring['metrics'] = metrics
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_peers(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get peers of specific symbol"
    symbol: Symbol to query for data.  Separating  by comma to query multiple symbols at once, ex : aapl,tsla
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-peers"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_summary(symbols: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get summary information of specific symbol"
    symbols: Symbol to query for data.  Separating  by comma to query multiple symbols at once, ex : aapl,tsla
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-summary"
    querystring = {'symbols': symbols, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_financials(symbol: str, target_currency: str='USD', period_type: str='annual', statement_type: str='income-statement', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get financials for specific symbol"
    symbol: Symbol to query for data, only one is allowed at a time.
        target_currency: The currency code
        period_type: One of the following : annual|quarterly|ttm
        statement_type: One of the following : income-statement|balance-sheet|cash-flow-statement
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-financials"
    querystring = {'symbol': symbol, }
    if target_currency:
        querystring['target_currency'] = target_currency
    if period_type:
        querystring['period_type'] = period_type
    if statement_type:
        querystring['statement_type'] = statement_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_v2_get_momentum(symbols: str, fields: str='chgp3m,chgp6m,chgp9m,chgp1y,low52,high52', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get momentum of specific symbol"
    symbols: Symbol to query for data.  Separating  by comma to query multiple symbols at once, ex : aapl,tsla
        fields: One of the following : movAvg10d|movAvg50d|movAvg100d|movAvg200d|pClose10d|pClose50d|pClose100d|pClose200d|pWeekVolShares|low52|high52|chgp5d|chgp1m|chgp3m|chgp6m|chgp9m|chgpYtd|chgp1y|chgp3y|chgt3y|chgp5y|chgt5y|chgp10y|chgt10y|chgt1m|chgtYtd|chgt1y
Separated by comma for multiple options. Ex : chgp3m,chgp6m,chgp9m,chgp1y,low52,high52,movAvg10d
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/v2/get-momentum"
    querystring = {'symbols': symbols, }
    if fields:
        querystring['fields'] = fields
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_analyst_recommendations(ticker_ids: str, group_by_month: bool=None, return_window: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Wall Street analyst recommendations for specific symbol"
    ticker_ids: The value of id field returned in .../symbols/get-meta-data
        group_by_month: Whether or not the data is grouped by month
        return_window: The return window
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-analyst-recommendations"
    querystring = {'ticker_ids': ticker_ids, }
    if group_by_month:
        querystring['group_by_month'] = group_by_month
    if return_window:
        querystring['return_window'] = return_window
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_holdings(symbols: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information  in Holdings tab of specific symbol"
    symbols: Symbol to query for data.  Separating  by comma to query multiple symbols at once, ex : aapl,tsla
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-holdings"
    querystring = {'symbols': symbols, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_valuation(symbols: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get valuation of specific symbols"
    symbols: Symbol to query for data.  Separating  by comma to query multiple symbols at once, ex : aapl,tsla
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-valuation"
    querystring = {'symbols': symbols, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transcripts_v2_get_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get transcript detail by id"
    id: The value of id returned in .../transcripts/v2/list endpoint
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/transcripts/v2/get-details"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_estimates(symbol: str, period_type: str='quarterly', data_type: str='revenues', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get estimated EPS/revenue of specific symbol by annual or quarterly"
    symbol: Symbol to query for data, only one is allowed at a time.
        period_type: One of the following : quarterly|annual
        data_type: One of the following : eps|revenues
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-estimates"
    querystring = {'symbol': symbol, }
    if period_type:
        querystring['period_type'] = period_type
    if data_type:
        querystring['data_type'] = data_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_chart(symbol: str, period: str='1Y', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data to draw chart for specific symbol"
    symbol: Symbol to query for data, only one is allowed at a time.
        period: One of the following : 1D|5D|1M|6M|YTD|1Y|3Y|5Y|10Y|MAX
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-chart"
    querystring = {'symbol': symbol, }
    if period:
        querystring['period'] = period
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_realtime_prices_deprecated(symbols: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "* Use .../market/get-realtime-quotes instead.
		Get real time prices"
    symbols: Query for prices in real time.  Separating  by comma to query multiple symbols at once, ex : aapl,tsla
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/market/get-realtime-prices"
    querystring = {'symbols': symbols, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_equity(filtercategory: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get different market equities"
    filtercategory: One of the following : us-equity-markets|us-equity-sectors|us-equity-factors|global-equity|countries-equity
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/market/get-equity"
    querystring = {'filterCategory': filtercategory, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_dividend_investing(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get dividend investing"
    
    """
    url = f"https://seeking-alpha.p.rapidapi.com/market/get-dividend-investing"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_realtime_quotes(sa_ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get real time quotes"
    sa_ids: The value of 'id' field returned in .../v2/auto-complete endpoint. This endpoint helps to query for real time quotes.  Separating  by comma to query multiple IDs at once. Ex : 612888,16123
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/market/get-realtime-quotes"
    querystring = {'sa_ids': sa_ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_deprecated(is_id: str, size: int=20, until: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List news of specific symbol"
    id: Symbol to query for data, only one is allowed at a time.
        size: The number of items per response
        until: The value of meta/page/minmaxPublishOn/min json object returned right in this endpoint to load next page
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/news/list"
    querystring = {'id': is_id, }
    if size:
        querystring['size'] = size
    if until:
        querystring['until'] = until
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transcripts_list_deprecated(is_id: str, until: int=None, size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List transcripts of specific symbol"
    id: Symbol to query for data, only one is allowed at a time.
        until: The value of meta/page/minmaxPublishOn/min json object returned right in this endpoint to load next page
        size: The number of items per response
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/transcripts/list"
    querystring = {'id': is_id, }
    if until:
        querystring['until'] = until
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles_list_deprecated(category: str, until: int=None, size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List articles by category"
    category: One of the following : etfs-and-funds|latest-articles|stock-ideas|editors-picks|stock-ideas::editors-picks|dividends|investing-strategy|dividends::reits|podcast|market-outlook
        until: The value of meta/page/minmaxPublishOn/min json object returned right in this endpoint to load next page
        size: The number of items per response
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/articles/list"
    querystring = {'category': category, }
    if until:
        querystring['until'] = until
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles_v2_list(until: int=None, since: int=None, number: int=1, size: int=20, category: str='latest-articles', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List articles by category"
    until: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'since' parameter to filter data by date range
        since: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'until' parameter to filter data by date range
        number: Page index for paging purpose
        size: The number of items per response (max 40)
        category: One of the following : editors-picks|latest-articles|dividends|dividends::dividend-ideas|dividends::dividend-quick-picks|dividends::dividend-strategy|dividends::reits|education::401k|education::cryptocurrency|education::dividends|education::etf|education::investing|education::portfolio-management|etfs-and-funds|etfs-and-funds::closed-end-funds|etfs-and-funds::etf-analysis|etfs-and-funds::mutual-funds|investing-strategy|investing-strategy::fixed-income|investing-strategy::portfolio-strategy|investing-strategy::retirement|market-outlook|market-outlook::commodities|market-outlook::cryptocurrency|market-outlook::economy|market-outlook::forex|market-outlook::gold-and-precious-metals|market-outlook::todays-market|sectors::communication-services|sectors::consumer-staples|sectors::energy|sectors::real-estate|sectors::real-estate|stock-ideas|stock-ideas::basic-materials|stock-ideas::consumer-goods|stock-ideas::financial|stock-ideas::healthcare|stock-ideas::industrial-goods|stock-ideas::ipos|stock-ideas::long-ideas|stock-ideas::quick-picks|stock-ideas::technology|stock-ideas::utilities
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/articles/v2/list"
    querystring = {}
    if until:
        querystring['until'] = until
    if since:
        querystring['since'] = since
    if number:
        querystring['number'] = number
    if size:
        querystring['size'] = size
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def analysis_v2_list(is_id: str, size: int=20, number: int=1, since: int=None, until: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List analysis of specific symbol"
    id: Symbol to query for data, only one is allowed at a time.
        size: The number of items per response (max 40)
        number: Page index for paging purpose
        since: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'until' parameter to filter data by date range
        until: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'since' parameter to filter data by date range
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/analysis/v2/list"
    querystring = {'id': is_id, }
    if size:
        querystring['size'] = size
    if number:
        querystring['number'] = number
    if since:
        querystring['since'] = since
    if until:
        querystring['until'] = until
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_v2_list_by_symbol(is_id: str, size: int=20, since: int=None, number: int=1, category: str=None, until: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List news by symbol"
    id: Symbol to query for data, only one is allowed at a time
        size: The number of items per response (max 40)
        since: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'until' parameter to filter data by date range
        number: Page index for paging purpose
        category: One of the following : dividend&#95;news|earnings&#95;news|m&#95;n&#95;a&#95;news
        until: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'since' parameter to filter data by date range
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/news/v2/list-by-symbol"
    querystring = {'id': is_id, }
    if size:
        querystring['size'] = size
    if since:
        querystring['since'] = since
    if number:
        querystring['number'] = number
    if category:
        querystring['category'] = category
    if until:
        querystring['until'] = until
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_v2_list(category: str, since: int=None, until: int=None, size: int=20, number: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List news by category"
    category: One of the following : market-news::all|market-news::top-news|market-news::on-the-move|market-news::market-pulse|market-news::notable-calls|market-news::buybacks|market-news::commodities|market-news::crypto|market-news::issuance|market-news::dividend-stocks|market-news::dividend-funds|market-news::earnings|earnings::earnings-news|market-news::global|market-news::guidance|market-news::ipos|market-news::spacs|market-news::politics|market-news::m-a|market-news::us-economy|market-news::consumer|market-news::energy|market-news::financials|market-news::healthcare|market-news::mlps|market-news::reits|market-news::technology
        since: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'until' parameter to filter data by date range
        until: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'since' parameter to filter data by date range
        size: The number of items per response (max 40)
        number: Page index for paging purpose
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/news/v2/list"
    querystring = {'category': category, }
    if since:
        querystring['since'] = since
    if until:
        querystring['until'] = until
    if size:
        querystring['size'] = size
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_day_watch(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market day watch"
    
    """
    url = f"https://seeking-alpha.p.rapidapi.com/market/get-day-watch"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_market_open(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market open"
    
    """
    url = f"https://seeking-alpha.p.rapidapi.com/market/get-market-open"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transcripts_get_details_deprecating(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get transcript detail by id
		* This endpoint is deprecating. Use .../transcripts/v2/get-details instead"
    id: The value of id returned in .../transcripts/list endpoint
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/transcripts/get-details"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def analysis_get_details_deprecating(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get analysis detail by id
		* This endpoint is deprecating. Use .../analysis/v2/get-details instead"
    id: The value of id returned in .../analysis/list endpoint
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/analysis/get-details"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def analysis_list_deprecated(is_id: str, size: int=20, until: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List analysis of specific symbol"
    id: Symbol to query for data, only one is allowed at a time.
        size: The number of items per response
        until: The value of meta/page/minmaxPublishOn/min json object returned right in this endpoint to load next page
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/analysis/list"
    querystring = {'id': is_id, }
    if size:
        querystring['size'] = size
    if until:
        querystring['until'] = until
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def analysis_v2_get_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get analysis detail by id"
    id: The value of id returned in .../analysis/list endpoint
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/analysis/v2/get-details"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screener_filters_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List available filters to be used in .../screeners/get-results endpoint"
    
    """
    url = f"https://seeking-alpha.p.rapidapi.com/screener-filters/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screeners_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all screeners (Top Rated Stocks, Top Quant Dividend Stocks,  Top Yield Monsters, etc...)"
    
    """
    url = f"https://seeking-alpha.p.rapidapi.com/screeners/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments_get_contents(is_id: int, comment_ids: int, sort: str='-top_parent_id', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get many comment's content at once by comment ids."
    id: The value of id returned in .../articles/list or .../articles/list-trending or .../articles/list-wall-street-breakfast endpoints
        comment_ids: The value of id field returned in .../comments/v2/list endpoint. You may pass this parameter multiple times to get content of many comments at once. Ex : ...&comment_ids=90666350&comment_ids=90666780&...
        sort: Order by newest : -top&#95;parent&#95;id | Order by oldest : leave empty
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/comments/get-contents"
    querystring = {'id': is_id, 'comment_ids': comment_ids, }
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments_v2_list(is_id: int, sort: str='-top_parent_id', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all comments relating to a post or article or news. This endpoint only returns site map structure of comments relating to a post. You need to use along with .../comments/get-contents endpoint to get the contents."
    id: The value of id returned in .../articles/list or .../articles/list-trending or .../articles/list-wall-street-breakfast endpoints
        sort: Order by newest : -top&#95;parent&#95;id | Order by oldest : leave empty
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/comments/v2/list"
    querystring = {'id': is_id, }
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_quant_rating_histories_deprecated(symbol: str, number: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get quant rating histories for specific symbol"
    symbol: Symbol to query for data, only one is allowed at a time.
        number: For paging purpose
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-quant-rating-histories"
    querystring = {'symbol': symbol, }
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments_get_sub_comments(is_id: int, sort: str='-top_parent_id', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is used to get sub or nested comments of another comment"
    id: The value of id returned in .../articles/list or .../articles/list-trending or .../articles/list-wall-street-breakfast endpoints
        sort: Order by newest : -top&#95;parent&#95;id | Order by oldest : leave empty
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/comments/get-sub-comments"
    querystring = {'id': is_id, }
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def screeners_detail(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get more information of a screener"
    id: The value of id field returned in .../screeners/list endpoint
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/screeners/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def press_releases_list_deprecated(is_id: str, size: int=20, until: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List press releases of specific symbol"
    id: Symbol to query for data, only one is allowed at a time.
        size: The number of items per response
        until: The value of meta/page/minmaxPublishOn/min json object returned right in this endpoint to load next page
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/press-releases/list"
    querystring = {'id': is_id, }
    if size:
        querystring['size'] = size
    if until:
        querystring['until'] = until
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def press_releases_get_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get press release detail by id"
    id: The value of id returned in .../press-releases/list endpoint
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/press-releases/get-details"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles_list_wall_street_breakfast(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List articles by category"
    
    """
    url = f"https://seeking-alpha.p.rapidapi.com/articles/list-wall-street-breakfast"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles_list_trending_deprecated(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List trending articles"
    
    """
    url = f"https://seeking-alpha.p.rapidapi.com/articles/list-trending"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles_v2_list_trending(since: int=None, until: int=None, size: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List trending articles"
    since: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'until' parameter to filter data by date range
        until: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'since' parameter to filter data by date range
        size: The number of items per response (max 40)
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/articles/v2/list-trending"
    querystring = {}
    if since:
        querystring['since'] = since
    if until:
        querystring['until'] = until
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_v2_list_trending(size: int=20, since: int=None, until: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List trending news"
    size: The number of items per response (max 40)
        since: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'until' parameter to filter data by date range
        until: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'since' parameter to filter data by date range
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/news/v2/list-trending"
    querystring = {}
    if size:
        querystring['size'] = size
    if since:
        querystring['since'] = since
    if until:
        querystring['until'] = until
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transcripts_v2_list(is_id: str, size: int=20, number: int=1, until: int=None, since: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List transcripts of specific symbol"
    id: Symbol to query for data, only one is allowed at a time.
        size: The number of items per response (max 40)
        number: Page index for paging purpose
        until: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'since' parameter to filter data by date range
        since: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'until' parameter to filter data by date range
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/transcripts/v2/list"
    querystring = {'id': is_id, }
    if size:
        querystring['size'] = size
    if number:
        querystring['number'] = number
    if until:
        querystring['until'] = until
    if since:
        querystring['since'] = since
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def press_releases_v2_list(is_id: str, size: int=20, number: int=1, until: int=None, since: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List press releases of specific symbol"
    id: Symbol to query for data, only one is allowed at a time.
        size: The number of items per response (max 40)
        number: Page index for paging purpose
        until: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'since' parameter to filter data by date range
        since: Unix timestamp (Epoch timestamp), ex : 1636693199
Maybe use together with 'until' parameter to filter data by date range
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/press-releases/v2/list"
    querystring = {'id': is_id, }
    if size:
        querystring['size'] = size
    if number:
        querystring['number'] = number
    if until:
        querystring['until'] = until
    if since:
        querystring['since'] = since
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete_deprecated(term: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggested symbols, authors, etc... by provided word or phrase"
    term: Any word or phrase that you are familiar with
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/auto-complete"
    querystring = {'term': term, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_auto_complete(query: str, type: str='people,symbols,pages', size: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggested symbols, authors, etc... by provided word or phrase"
    query: Any word or phrase that you are familiar with
        type: One of the following : people|symbols|pages. Separated by comma for multiple options
        size: The number items per response (10 is maximum)
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/v2/auto-complete"
    querystring = {'query': query, }
    if type:
        querystring['type'] = type
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments_list_deprecated(is_id: int, parent_count: int=20, sort: str='-top_parent_id', from_id: int=88004158, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all comments relating to a post or article or news"
    id: The value of id returned in .../articles/list or .../articles/list-trending or .../articles/list-wall-street-breakfast endpoints
        parent_count: For paging purpose (max 20)
        sort: Order by newest : -top_parent_id | Order by oldest : leave empty
        from_id: Leave empty to load the first page or get suitable value of the last comment id returned right in this endpoint with parentId being \"null\" to load the next page
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/comments/list"
    querystring = {'id': is_id, }
    if parent_count:
        querystring['parent_count'] = parent_count
    if sort:
        querystring['sort'] = sort
    if from_id:
        querystring['from_id'] = from_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def authors_get_details(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get author details"
    slug: The value of people/slug json object returned in .../auto-complete endpoint
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/authors/get-details"
    querystring = {'slug': slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_get_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get analysis detail by id"
    id: The value of id returned in .../news/list or .../news/list-trending endpoint
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/news/get-details"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols_get_analyst_ratings_deprecated(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Wall Street analyst ratings for specific symbol"
    symbol: Symbol to query for data, only one is allowed at a time.
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/symbols/get-analyst-ratings"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles_get_details(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get analysis detail by id"
    id: The value of id returned in .../articles/list or .../articles/list-trending or .../articles/list-wall-street-breakfast endpoints
        
    """
    url = f"https://seeking-alpha.p.rapidapi.com/articles/get-details"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_trending_deprecated(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest trending news"
    
    """
    url = f"https://seeking-alpha.p.rapidapi.com/news/list-trending"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

