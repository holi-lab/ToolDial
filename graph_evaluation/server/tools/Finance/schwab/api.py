import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def quote_get_chart(enddate: str, startdate: str, symbol: str, dataperiod: str='Hour', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data to draw chart"
    enddate: The format is yyyy-MM-dd'T'HH:mm:ss
        startdate: The format is yyyy-MM-dd'T'HH:mm:ss
        symbol: The symbol to get information
        dataperiod: One of the following : Minute|Hour|Day|Week|Month
        
    """
    url = f"https://schwab.p.rapidapi.com/quote/get-chart"
    querystring = {'endDate': enddate, 'startDate': startdate, 'symbol': symbol, }
    if dataperiod:
        querystring['dataPeriod'] = dataperiod
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quote_get_option_chains(root: str, symbol: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get option chains"
    root: The value of RootSymbol field returned in .../quote/get-option-fundamentals endpoint.
        symbol: The symbol to get details information
        date: The format is yyyyMMdd
        
    """
    url = f"https://schwab.p.rapidapi.com/quote/get-option-chains"
    querystring = {'root': root, 'symbol': symbol, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quote_get_summary(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get summary information of quote"
    symbol: The symbol to get information
        
    """
    url = f"https://schwab.p.rapidapi.com/quote/get-summary"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_volatility(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market volatility
		* There are response images in encoded base 64 string, you need to decode to get the images yourself"
    
    """
    url = f"https://schwab.p.rapidapi.com/market/get-volatility"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_get_market_update_audio(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get audio link to hear market update news"
    
    """
    url = f"https://schwab.p.rapidapi.com/news/get-market-update-audio"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quote_get_details(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get details information of quote"
    symbol: The symbol to get details information
        
    """
    url = f"https://schwab.p.rapidapi.com/quote/get-details"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_get_market_update(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get latest market update news"
    
    """
    url = f"https://schwab.p.rapidapi.com/news/get-market-update"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(matchchars: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestion by word or phase"
    
    """
    url = f"https://schwab.p.rapidapi.com/auto-complete"
    querystring = {'MatchChars': matchchars, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_reports(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reports about the market
		* You need to use .../content/decrypt endpoint to decrypt content returned by Url fields."
    
    """
    url = f"https://schwab.p.rapidapi.com/market/get-reports"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quote_get_margin_requirements(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get margin requirements"
    symbol: The symbol to get details information
        
    """
    url = f"https://schwab.p.rapidapi.com/quote/get-margin-requirements"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_sectors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brief information about all sectors"
    
    """
    url = f"https://schwab.p.rapidapi.com/market/get-sectors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_get_details(docid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Read news in details"
    docid: The value of DocumentID returned in .../news/list-latest endpoint.
        
    """
    url = f"https://schwab.p.rapidapi.com/news/get-details"
    querystring = {'docID': docid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_indices(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available market indices"
    
    """
    url = f"https://schwab.p.rapidapi.com/market/get-indices"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quote_get_option_fundamentals(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get option fundamentals of quote"
    symbol: The symbol to get information
        
    """
    url = f"https://schwab.p.rapidapi.com/quote/get-option-fundamentals"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_get_market_update_details(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Read market update news in details"
    
    """
    url = f"https://schwab.p.rapidapi.com/news/get-market-update-details"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_top_mentions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List top mentions stock quotes"
    
    """
    url = f"https://schwab.p.rapidapi.com/market/get-top-mentions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_latest(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List latest news"
    
    """
    url = f"https://schwab.p.rapidapi.com/news/list-latest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_movers(ranktype: str, exchange: str, sectorcusip: str='ALL', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List recent movers in the market"
    ranktype: One of the following : MostActives|PctChgGainers|PctChgLosers|NetGainers|NetLosers|52WkHigh|52WkLow
        exchange: One of the following : US|USN|USQ|USA
        sectorcusip: The value of Sectors/SectorCusip returned right in this endpoint.
        
    """
    url = f"https://schwab.p.rapidapi.com/market/get-movers"
    querystring = {'rankType': ranktype, 'exchange': exchange, }
    if sectorcusip:
        querystring['sectorCusip'] = sectorcusip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def content_decrypt(encryptedcontent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Used to decrypt content of Url fields returned in .../market/get-reports endpoint"
    encryptedcontent: The value of Url fields returned in .../market/get-reports endpoint. Ex : "Url": "LZ9K2mvPqP3WhvDsQw09pWbKJfLyyhO630sWnBlegzL7VYQyVA3Q4RKXGO2x%2F0SdXvXBYRH684q4xUo9H4uubq9KJNvdHRO3KLBxTvnKzaU%3D"
        
    """
    url = f"https://schwab.p.rapidapi.com/content/decrypt"
    querystring = {'encryptedContent': encryptedcontent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_futures(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get future reports about the market
		* There are response images in encoded base 64 string, you need to decode to get the images yourself"
    
    """
    url = f"https://schwab.p.rapidapi.com/market/get-futures"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "schwab.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

