import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def news_list_by_portfolio(tickerids: str, currentnewsid: int=0, pagesize: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List news by multiple ticker id"
    tickerids: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc...Separated by comma for multiple tickers
        currentnewsid: For paging purpose, the last news id returned right in this endpoint, pass 0 for the first query
        pagesize: For paging purpose, maximum is 20
        
    """
    url = f"https://webull.p.rapidapi.com/news/list-by-portfolio"
    querystring = {'tickerIds': tickerids, }
    if currentnewsid:
        querystring['currentNewsId'] = currentnewsid
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_ipo_center(regionid: int, status: str='filing', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get IPO center information of specific region"
    regionid: The value of id field returned in .../regions/list endpoint
        status: One of the following : filing|pricing|buying
        
    """
    url = f"https://webull.p.rapidapi.com/market/get-ipo-center"
    querystring = {'regionId': regionid, }
    if status:
        querystring['status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_top_gainers(regionid: int, pagesize: int=20, ranktype: str='1d', pageindex: int=1, direction: int=-1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market top gainers in specific region"
    regionid: The value of id field returned in .../regions/list endpoint
        pagesize: For paging purpose, maximum is 20
        ranktype: One of the following : preMarket|afterMarket|5min|1d|5d|1m|3m|52w
        pageindex: For paging purpose
        direction: The order direction -1 | 1
        
    """
    url = f"https://webull.p.rapidapi.com/market/get-top-gainers"
    querystring = {'regionId': regionid, }
    if pagesize:
        querystring['pageSize'] = pagesize
    if ranktype:
        querystring['rankType'] = ranktype
    if pageindex:
        querystring['pageIndex'] = pageindex
    if direction:
        querystring['direction'] = direction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_top_active(regionid: int, pageindex: int=1, ranktype: str='volume', pagesize: int=20, direction: int=-1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market top active in specific region"
    regionid: The value of id field returned in .../regions/list endpoint
        pageindex: For paging purpose
        ranktype: One of the following : volume|range|turnoverRatio
        pagesize: For paging purpose, maximum is 20
        direction: The order direction -1 | 1
        
    """
    url = f"https://webull.p.rapidapi.com/market/get-top-active"
    querystring = {'regionId': regionid, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    if ranktype:
        querystring['rankType'] = ranktype
    if pagesize:
        querystring['pageSize'] = pagesize
    if direction:
        querystring['direction'] = direction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_common_ranking(regionid: int, pageindex: int=1, pagesize: int=20, ranktype: str='hkMainBoard', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market common ranking in specific region"
    regionid: The value of id field returned in .../regions/list endpoint
        pageindex: For paging purpose
        pagesize: For paging purpose, maximum is 20
        ranktype: One of the following : hkMainBoard|hkGem|hkStocks|hotStocks
        
    """
    url = f"https://webull.p.rapidapi.com/market/get-common-ranking"
    querystring = {'regionId': regionid, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    if pagesize:
        querystring['pageSize'] = pagesize
    if ranktype:
        querystring['rankType'] = ranktype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_earnings(regionid: int, pageindex: int=1, startdate: str=None, pagesize: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market earnings in specific region"
    regionid: The value of id field returned in .../regions/list endpoint
        pageindex: For paging purpose
        startdate: The format date is yyyy-MM-dd. Ex : 2021-08-10
        pagesize: For paging purpose, maximum is 20
        
    """
    url = f"https://webull.p.rapidapi.com/market/get-earnings"
    querystring = {'regionId': regionid, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    if startdate:
        querystring['startDate'] = startdate
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def regions_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all supported regions"
    
    """
    url = f"https://webull.p.rapidapi.com/regions/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_summary(regionid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market summary at request time for specific region"
    regionid: The value of id field returned in .../regions/list endpoint
        
    """
    url = f"https://webull.p.rapidapi.com/market/get-summary"
    querystring = {'regionId': regionid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_dividends(regionid: int, pagesize: int=20, pageindex: int=1, startdate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market dividends in specific region"
    regionid: The value of id field returned in .../regions/list endpoint
        pagesize: For paging purpose, maximum is 20
        pageindex: For paging purpose
        startdate: The format date is yyyy-MM-dd. Ex : 2021-08-10
        
    """
    url = f"https://webull.p.rapidapi.com/market/get-dividends"
    querystring = {'regionId': regionid, }
    if pagesize:
        querystring['pageSize'] = pagesize
    if pageindex:
        querystring['pageIndex'] = pageindex
    if startdate:
        querystring['startDate'] = startdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_splits(regionid: int, startdate: str=None, pageindex: int=1, pagesize: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market splits in specific region"
    regionid: The value of id field returned in .../regions/list endpoint
        startdate: The format date is yyyy-MM-dd. Ex : 2021-08-10
        pageindex: For paging purpose
        pagesize: For paging purpose, maximum is 20
        
    """
    url = f"https://webull.p.rapidapi.com/market/get-splits"
    querystring = {'regionId': regionid, }
    if startdate:
        querystring['startDate'] = startdate
    if pageindex:
        querystring['pageIndex'] = pageindex
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_realtime_quotes(ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get multiple stock quotes in real time."
    ids: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... Separated by comma for multiple tickers
        
    """
    url = f"https://webull.p.rapidapi.com/market/get-realtime-quotes"
    querystring = {'ids': ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def industries_get_performing_detail(regionid: int, groupid: str, pagesize: int=20, pageindex: int=1, direction: int=-1, industrytype: str='today', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List industry performing detail"
    regionid: The value of id field returned in .../regions/list endpoint
        groupid: The value of id field returned in .../industries/list-best-performing endpoint
        pagesize: For paging purpose, maximum is 20
        pageindex: For paging purpose
        direction: The order direction -1(desc) | 1(asc)
        industrytype: One of the following : today|5d|1m|3m
        
    """
    url = f"https://webull.p.rapidapi.com/industries/get-performing-detail"
    querystring = {'regionId': regionid, 'groupId': groupid, }
    if pagesize:
        querystring['pageSize'] = pagesize
    if pageindex:
        querystring['pageIndex'] = pageindex
    if direction:
        querystring['direction'] = direction
    if industrytype:
        querystring['industryType'] = industrytype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_charts(tickerids: str, count: int=400, direction: int=-1, type: str='m5', timestamp: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get multiple history pricing  charts"
    tickerids: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... Separated by comma for multiple tickers
        count: The number of items per ticker, for paging purpose. Maximum is 800
        direction: The order direction -1(desc) | 1(asc)
        type: The interval of time, one of the following : m1(1 min)|m5(5 mins)|m15(15 mins)|m30(30 mins)|m60(1 hour)|m120(2 hours)|m240(4 hours)|d1(1 day)|w1(1 weekmth1(1 month)|mth3(3 months)|y1(1 year)
        timestamp: For paging purpose, load data before this timestamp (in seconds). Ex : 1629122700
        
    """
    url = f"https://webull.p.rapidapi.com/market/get-charts"
    querystring = {'tickerIds': tickerids, }
    if count:
        querystring['count'] = count
    if direction:
        querystring['direction'] = direction
    if type:
        querystring['type'] = type
    if timestamp:
        querystring['timestamp'] = timestamp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_sparks(ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to draw small charts which often display next to symbols"
    ids: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... Separated by comma for multiple tickers
        
    """
    url = f"https://webull.p.rapidapi.com/market/get-sparks"
    querystring = {'ids': ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def industries_list_best_performing(regionid: int, industrytype: str='today', pagesize: int=20, pageindex: int=1, direction: int=-1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List best performing industries"
    regionid: The value of id field returned in .../regions/list endpoint
        industrytype: One of the following : today|5d|1m|3m
        pagesize: For paging purpose, maximum is 20
        pageindex: For paging purpose
        direction: The order direction -1(desc) | 1(asc)
        
    """
    url = f"https://webull.p.rapidapi.com/industries/list-best-performing"
    querystring = {'regionId': regionid, }
    if industrytype:
        querystring['industryType'] = industrytype
    if pagesize:
        querystring['pageSize'] = pagesize
    if pageindex:
        querystring['pageIndex'] = pageindex
    if direction:
        querystring['direction'] = direction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_search(keyword: str, pageindex: int=1, pagesize: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for stock, index, fund, etc... by terms or phrase"
    keyword: Any term or phrase that you are familiar with
        pageindex: For paging purpose
        pagesize: For paging purpose, maximum is 20
        
    """
    url = f"https://webull.p.rapidapi.com/stock/search"
    querystring = {'keyword': keyword, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_realtime_quote(tickerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock quote in real time."
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-realtime-quote"
    querystring = {'tickerId': tickerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_trending_chart(tickerid: str, trendtype: str='m1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brief pricing history chart in a period of time"
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        trendtype: One of the following : m1|m3|m6|y1|y5|all
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-trending-chart"
    querystring = {'tickerId': tickerid, }
    if trendtype:
        querystring['trendType'] = trendtype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_options(tickerid: str, unsymbol: str, expiredate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock options"
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        unsymbol: The symbol relating to tickerId
        expiredate: The format date is yyyy-MM-dd. Ex : 2021-09-03
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-options"
    querystring = {'tickerId': tickerid, 'unSymbol': unsymbol, }
    if expiredate:
        querystring['expireDate'] = expiredate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_rating_pricing_target(tickerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock rating and pricing target"
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-rating-pricing-target"
    querystring = {'tickerId': tickerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_short_interest(tickerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock short interest"
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-short-interest"
    querystring = {'tickerId': tickerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_cost_distribution_analysis(tickerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock cost distribution and analysis"
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-cost-distribution-analysis"
    querystring = {'tickerId': tickerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_peers(tickerid: str, type: int=2, hasnum: int=0, pagesize: int=20, direction: int=-1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock peers or ETF"
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        type: One of the following : 2 (Peers) | 3 (ETF)
        hasnum: The offset for paging purpose
        pagesize: For paging purpose, maximum is 20
        direction: The order direction -1(desc) | 1(asc)
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-peers"
    querystring = {'tickerId': tickerid, }
    if type:
        querystring['type'] = type
    if hasnum:
        querystring['hasNum'] = hasnum
    if pagesize:
        querystring['pageSize'] = pagesize
    if direction:
        querystring['direction'] = direction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_industry_sector(tickerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get relating industry sector"
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-industry-sector"
    querystring = {'tickerId': tickerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_financials(fiscalperiod: int, tickerid: int, fiscalyear: int, type: int=2, count: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get relating company financials"
    fiscalperiod: The fiscal period, from 1 to 4
        tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        fiscalyear: The fiscal year
        type: One of the following : 1 (Annual) | 2 (Quarterly)
        count: The number of items returned
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-financials"
    querystring = {'fiscalPeriod': fiscalperiod, 'tickerId': tickerid, 'fiscalYear': fiscalyear, }
    if type:
        querystring['type'] = type
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_company_profile(tickerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get relating company profile"
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-company-profile"
    querystring = {'tickerId': tickerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_performance(tickerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock fund performance"
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-performance"
    querystring = {'tickerId': tickerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_fund_profile(tickerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock fund profile"
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-fund-profile"
    querystring = {'tickerId': tickerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_asset_allocation(tickerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock fund asset allocation"
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-asset-allocation"
    querystring = {'tickerId': tickerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_stat(tickerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock statistic"
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-stat"
    querystring = {'tickerId': tickerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_press_releases(tickerid: int, lastannouncementid: int=None, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get press releases"
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        lastannouncementid: For paging purpose, the last announcement Id returned right in this endpoint, leave empty for the first query.
        limit: For paging purpose, the number of items per response
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-press-releases"
    querystring = {'tickerId': tickerid, }
    if lastannouncementid:
        querystring['lastAnnouncementId'] = lastannouncementid
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_get_dividends_splits(tickerid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stock dividends and splits"
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        
    """
    url = f"https://webull.p.rapidapi.com/stock/get-dividends-splits"
    querystring = {'tickerId': tickerid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_search(keyword: str, pageindex: int=1, pagesize: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for news by terms or phrase"
    keyword: Any term or phrase that you are familiar with
        pageindex: For paging purpose
        pagesize: For paging purpose, maximum is 20
        
    """
    url = f"https://webull.p.rapidapi.com/news/search"
    querystring = {'keyword': keyword, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_top(pagesize: int=20, currentnewsid: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List top news"
    pagesize: For paging purpose, maximum is 20
        currentnewsid: For paging purpose, the last news id returned right in this endpoint, pass 0 for the first query.
        
    """
    url = f"https://webull.p.rapidapi.com/news/list-top"
    querystring = {}
    if pagesize:
        querystring['pageSize'] = pagesize
    if currentnewsid:
        querystring['currentNewsId'] = currentnewsid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_by_ticker(tickerid: int, pagesize: int=20, currentnewsid: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List news by ticker id"
    tickerid: The value of tickerId field returned in other endpoints, such as .../stock/search or .../market/get-common-ranking or .../market/get-top-active or .../market/get-top-gainers or etc... 
        pagesize: For paging purpose, maximum is 20
        currentnewsid: For paging purpose, the last news id returned right in this endpoint, pass 0 for the first query
        
    """
    url = f"https://webull.p.rapidapi.com/news/list-by-ticker"
    querystring = {'tickerId': tickerid, }
    if pagesize:
        querystring['pageSize'] = pagesize
    if currentnewsid:
        querystring['currentNewsId'] = currentnewsid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_flash(currentnewsid: int=0, pagesize: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List flash 27/4 news"
    currentnewsid: For paging purpose, the last news id returned right in this endpoint, pass 0 for the first query.
        pagesize: For paging purpose, maximum is 20
        
    """
    url = f"https://webull.p.rapidapi.com/news/list-flash"
    querystring = {}
    if currentnewsid:
        querystring['currentNewsId'] = currentnewsid
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currencies_list(userregionid: int, pagesize: int=20, hasnum: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all global currencies and rates"
    userregionid: The value of id field returned in .../regions/list endpoint
        regionid: The value of id field returned in .../regions/list endpoint
        pagesize: For paging purpose, maximum is 20
        hasnum: The offset for paging purpose
        
    """
    url = f"https://webull.p.rapidapi.com/currencies/list"
    querystring = {'userRegionId': userregionid, }
    if pagesize:
        querystring['pageSize'] = pagesize
    if hasnum:
        querystring['hasNum'] = hasnum
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webull.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

