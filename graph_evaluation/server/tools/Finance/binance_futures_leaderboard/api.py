import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_other_performance(encrypteduid: str, tradetype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the trader performance"
    encrypteduid: The encrypted UID of the trader
        tradetype: The trade type to search. PERPETUAL is USDⓈ-M and DELIVERY is COIN-M
        
    """
    url = f"https://binance-futures-leaderboard1.p.rapidapi.com/v1/getOtherPerformance"
    querystring = {'encryptedUid': encrypteduid, }
    if tradetype:
        querystring['tradeType'] = tradetype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_nickname(nickname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the trader by nickname"
    nickname: The trader nickname to search
        
    """
    url = f"https://binance-futures-leaderboard1.p.rapidapi.com/v1/searchNickname"
    querystring = {'nickname': nickname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_trader_positions(encrypteduid: str, tradetype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trader open positions"
    encrypteduid: The encrypted UID of the trader
        tradetype: The trade type to search. PERPETUALis USDⓈ-M and DELIVERY is COIN-M
        
    """
    url = f"https://binance-futures-leaderboard1.p.rapidapi.com/v2/getTraderPositions"
    querystring = {'encryptedUid': encrypteduid, }
    if tradetype:
        querystring['tradeType'] = tradetype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_other_leaderboard_base_info(encrypteduid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the trader information"
    encrypteduid: The encrypted UID of the trader
        
    """
    url = f"https://binance-futures-leaderboard1.p.rapidapi.com/v1/getOtherLeaderboardBaseInfo"
    querystring = {'encryptedUid': encrypteduid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_trader_info(encrypteduid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trader information"
    encrypteduid: The encrypted UID of the trader
        
    """
    url = f"https://binance-futures-leaderboard1.p.rapidapi.com/v2/getTraderInfo"
    querystring = {'encryptedUid': encrypteduid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_other_trade_record(encrypteduid: str, traderecordtype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the other trade record"
    encrypteduid: The encrypted UID of the trader
        traderecordtype: The trader's trade record type
        
    """
    url = f"https://binance-futures-leaderboard1.p.rapidapi.com/v1/getOtherTradeRecord"
    querystring = {'encryptedUid': encrypteduid, }
    if traderecordtype:
        querystring['tradeRecordType'] = traderecordtype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_trader(nickname: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for traders by nickname"
    nickname: The trader nickname to search
        
    """
    url = f"https://binance-futures-leaderboard1.p.rapidapi.com/v2/searchTrader"
    querystring = {'nickname': nickname, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_leaderboard_rank(statisticstype: str=None, isshared: bool=None, tradetype: str=None, periodtype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the leaderboard rank"
    statisticstype: The statistics type to search
        isshared: Include shared positions
        tradetype: The trade type to search. PERPETUAL is USDⓈ-M and DELIVERY is COIN-M
        periodtype: The period type to search
        
    """
    url = f"https://binance-futures-leaderboard1.p.rapidapi.com/v1/getLeaderboardRank"
    querystring = {}
    if statisticstype:
        querystring['statisticsType'] = statisticstype
    if isshared:
        querystring['isShared'] = isshared
    if tradetype:
        querystring['tradeType'] = tradetype
    if periodtype:
        querystring['periodType'] = periodtype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_other_position(encrypteduid: str, tradetype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trader open positions"
    encrypteduid: The encrypted UID of the trader
        tradetype: The trade type to search. PERPETUAL is USDⓈ-M and DELIVERY is COIN-M
        
    """
    url = f"https://binance-futures-leaderboard1.p.rapidapi.com/v1/getOtherPosition"
    querystring = {'encryptedUid': encrypteduid, }
    if tradetype:
        querystring['tradeType'] = tradetype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_leaderboard(istrader: bool=None, statisticstype: str=None, periodtype: str=None, isshared: bool=None, tradetype: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of traders on the leaderboard"
    istrader: Include only copytraders
        statisticstype: The statistics type to search
        periodtype: The period type to search
        isshared: Include shared positions
        tradetype: The trade type to search. PERPETUAL is USDⓈ-M and DELIVERY is COIN-M
        
    """
    url = f"https://binance-futures-leaderboard1.p.rapidapi.com/v2/searchLeaderboard"
    querystring = {}
    if istrader:
        querystring['isTrader'] = istrader
    if statisticstype:
        querystring['statisticsType'] = statisticstype
    if periodtype:
        querystring['periodType'] = periodtype
    if isshared:
        querystring['isShared'] = isshared
    if tradetype:
        querystring['tradeType'] = tradetype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "binance-futures-leaderboard1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

