import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def days_from_date(symbol: str, daysfrom: int, datefrom: str, month: int=7, offset: int=None, format: str='json', inpast: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns futures contracts for the date falling on the number of days from the date you specified. This is useful for getting contracts closest to a date without having to determine the last date the contract was traded.
		
		Consider wanting contracts for WTI Crude that were traded 30 days before Christmas 2021, with Thanksgiving in the US being November 25, which date were contracts last traded? Specifying the dateFrom parameter as "2021-12-25" and the daysFrom parameter as 30 removes the guess work and gives the contract less-than-or-equal-to 30 days from the <dateFrom> parameter.
		
		Use the optional <month> parameter to get a specific month contract.
		
		Use the optional <inPast> parameter to specify if <daysFrom> should be added or subtracted from <dateFrom>. Setting <inPast> to false will result in contracts that fall after the given date. Default is true.
		
		**✴ Return Limited: 100 Objects
		✴ Oldest Date: 2010-01-01**
		
		*This endpoint will only return 100 objects per request. Use the <offset> parameter to paginate results.*"
    
    """
    url = f"https://futures.p.rapidapi.com/days-from"
    querystring = {'symbol': symbol, 'daysFrom': daysfrom, 'dateFrom': datefrom, }
    if month:
        querystring['month'] = month
    if offset:
        querystring['offset'] = offset
    if format:
        querystring['format'] = format
    if inpast:
        querystring['inPast'] = inpast
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "futures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_series(is_from: str, symbol: str, offset: int=None, to: str='2022-01-30', format: str='json', month: str='8', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns futures contracts over a specified date range for a given symbol. Use this endpoint for historical data.
		
		The <to> parameter is optional. Not specifying this parameter will return all contracts between the given <from> parameter and the latest available date.
		
		Use the optional <month> parameter to get a specific month contract over the given period of time.
		
		**✴ Return Limited: 100 Objects
		✴ Oldest Date: 2010-01-01**
		
		*This endpoint will only return 100 objects per request. Use the <offset> parameter to paginate results.*"
    
    """
    url = f"https://futures.p.rapidapi.com/time-series"
    querystring = {'from': is_from, 'symbol': symbol, }
    if offset:
        querystring['offset'] = offset
    if to:
        querystring['to'] = to
    if format:
        querystring['format'] = format
    if month:
        querystring['month'] = month
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "futures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def supported_assets(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns all symbols supported by the * last *and* time-series *endpoint."
    
    """
    url = f"https://futures.p.rapidapi.com/symbols"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "futures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_contracts_for_symbol(symbol: str, month: int=7, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns each of the latest futures contracts for a given symbol for the next twelve months.
		
		Use the optional *<month>* parameter to return the latest contract for a given month."
    
    """
    url = f"https://futures.p.rapidapi.com/last"
    querystring = {'symbol': symbol, }
    if month:
        querystring['month'] = month
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "futures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

