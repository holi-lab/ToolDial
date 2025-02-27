import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_okx_p2p_ads(fiatcurrency: str, side: str, cryptocurrency: str, quoteminamountperorder: int=100, paymentmethod: str='SEPA+Instant', numberperpage: int=10, sorttype: str='price_desc', currentpage: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API will help you to get real-time buy and sell ads from OKX P2P exchange for any supported crypto and fiat currency pair."
    fiatcurrency: usd, eur, rub...
        side: buy or sell
        cryptocurrency: use tradable crypto ticker
        quoteminamountperorder: Use it if you want to limit orders by minimum amount
        paymentmethod: Use to limit the results by selected payment method
        sorttype: price_desc or price_asc
        
    """
    url = f"https://okx-p2p-api.p.rapidapi.com/"
    querystring = {'fiatCurrency': fiatcurrency, 'side': side, 'cryptoCurrency': cryptocurrency, }
    if quoteminamountperorder:
        querystring['quoteMinAmountPerOrder'] = quoteminamountperorder
    if paymentmethod:
        querystring['paymentMethod'] = paymentmethod
    if numberperpage:
        querystring['numberPerPage'] = numberperpage
    if sorttype:
        querystring['sortType'] = sorttype
    if currentpage:
        querystring['currentPage'] = currentpage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "okx-p2p-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

