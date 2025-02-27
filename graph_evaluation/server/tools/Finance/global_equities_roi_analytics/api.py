import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dynamic_query_for_global_equities_data(stockcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "An API that shows an equity's fundamental, ROI analysis and its long term dividend history
		
		You will get a comprehensive information on:
		
		Fundamentals of a Stock [Sector, Current Price, PE ratio, Cash Balance, Book Value, Nett Income, Total Debt, Dividend Yield]
		Dividend History [Get last 100 record of dividend history with Announcement Date and Dividend Amount]
		Dividend Analysis [Dividend Suspension Tracker, All Time Average Yield, All Time Average Dividend per Payout, Total Dividend Payment Count, Current Dividend Paying Status]
		Bonus: Company Logo in PNG format (Lightweight)
		Currently Supports NASDAQ, SGX and KLSE stock exchanges"
    stockcode: An API that shows an equity’s fundamental, ROI analysis and its long term dividend history

You will get a comprehensive information on:

Fundamentals of a Stock [Sector, Current Price, PE ratio, Cash Balance, Book Value, Nett Income, Total Debt, Dividend Yield]
Dividend History [Get last 100 record of dividend history with Announcement Date and Dividend Amount]
Dividend Analysis [Dividend Suspension Tracker, All Time Average Yield, All Time Average Dividend per Payout, Total Dividend Payment Count, Current Dividend Paying Status]
Bonus: Company Logo in PNG format (Lightweight)
Currently Supports NASDAQ, SGX and KLSE stock exchanges

To get API for Nasdaq (US) stocks:
• Simply type the target Stock’s ticker symbols or Company Name and append this string ‘.nasdaqusa into the Param’s field in the endpoint [eg: Microsoft.nasdaqusa or MSFT.nasdaqusa for Microsoft Inc | Apple.nasdaqusa or AAPL.nasdaqusa for Apple Inc]
• You can also refer this website to reconfirm the NASDAQ stock’s ticker symbols should you need assistance (https://www.nasdaq.com/market-activity/stocks/screener)

To get API for SGX (Singaporean) stocks:
• Simply type the target Stock’s ticker symbols and append this string ‘.sgxsingapore into the Param’s field in the endpoint [eg: Z74.sgxsingapore or SingTel.sgxsingapore for Singapore Telecommunications Limited | D05.sgxsingapore or DBS.sgxsingapore for DBS Group Holdings Ltd]
• You can also refer this website to reconfirm the SGX stock’s ticker symbols should you need assistance (https://www.sgx.com/securities/stock-screener)

To get API for KLSE (Malaysian) stocks:
• Simply type the target Stock’s ticker symbols and append this string ‘.klsemalaysia’ into the Param’s field in the endpoint [eg: Maybank.klsemalaysia or 1155.klsemalaysia for Malayan Banking Berhad | Genting.klsemalaysia or 3182.klsemalaysia for Genting Berhad]
• You can also refer this website to reconfirm the KLSE stock’s ticker symbols should you need assistance (https://www.bursamalaysia.com)
        
    """
    url = f"https://global-equities-roi-analytics.p.rapidapi.com/search/{stockcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-equities-roi-analytics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

