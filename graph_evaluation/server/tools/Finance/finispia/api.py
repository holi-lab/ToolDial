import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_compliance(symbol: str, exchange: str='NASDAQ', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Click in check button will display compliance of company"
    exchange: Default ( NASDAQ and NYSE)
**Available exchanges:**
- Abu Dhabi
- Aktietorget
- Amman
- Amsterdam
- Athens
- Bahrain
- BATS Europe
- Beirut
- Belgrade
- Berlin
- BIVA
- BM&FBovespa
- Bosnia
- Botswana
- Bratislava
- Brussels
- BRVM
- BSE
- Bucharest
- Budapest
- Buenos Aires
- Caracas
- Casablanca
- Colombia
- Colombo
- Copenhagen
- Costa Rica
- CSE
- Cyprus
- Dar Es Salaam
- Doha
- DSE
- Dubai
- Dusseldorf
- Egypt
- Frankfurt
- Hamburg
- Hanoi
- Helsinki
- Ho Chi Minh
- Hong Kong
- Iceland
- Ireland
- Istanbul
- ISX
- Jakarta
- Jamaica
- Johannesburg
- Karachi
- KASE
- Kenya
- KONEX
- KOSDAQ
- Kuala Lumpur
- Kuwait City
- Lagos
- LATIBEX
- Lima
- Lisbon
- Ljubljana
- London
- Luxembourg
- Madrid
- Malawi
- Malta
- Mauritius
- Mexico
- Milan
- MNSE
- Mongolia
- Moscow
- Munich
- Namibia
- NASDAQ
- NASDAQ OMX Riga
- NEO
- NGM
- NSE
- NYSE
- NYSE Amex
- Oman
- Oslo
- OTC Markets
- Paris
- Philippines
- Prague
- Ramallah
- Rwanda
- Santiago
- Saudi Arabia
- Seoul
- Shanghai
- Shenzhen
- Singapore
- Sofia
- SOMA
- Stockholm
- Stuttgart
- Switzerland
- Sydney
- Taiwan
- Tallinn
- Tel Aviv
- Thailand
- Tokyo
- Toronto
- TPEX
- TradeGate
- TSXV
- Tunis
- Uganda
- Ukraine
- Vienna
- Vilnius
- Warsaw
- Xetra
- Zagreb
- Zambia
- Zimbabwe
        
    """
    url = f"https://finispia2.p.rapidapi.com/company/compliance"
    querystring = {'symbol': symbol, }
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finispia2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

