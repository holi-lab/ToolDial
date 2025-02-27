import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def economical_events_list(page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a list of all Economical Events which have an high prioritization and therefore could effect markets."
    page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/economical/events/"
    querystring = {}
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earning_events_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a specific earning. F.ex. research for AAPL's earnings or the event ID."
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/earning/events/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dividends_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a specific dividend event"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/dividends/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dividends(page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all dividends that are upcoming/occurred recently"
    page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/dividends/"
    querystring = {}
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def earning_events_list(page_size: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all current earnings. We use data in an interval of the last month to the next year."
    page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/earning/events/"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def economical_events_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve a an Economical Event by its name or Event ID to inspect it in more detail."
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/economical/events/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datatables_retrieve(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Premium: Interact with your table by querying it with its unique ID. You can list, retrieve, delete, update, create and replace tables on the go."
    id: A unique integer value identifying this table.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/datatables/{is_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datatable_query_id_id(verbose: bool=None, camel: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint can be important to retrieve **as many data as you want within 1 request** . It queries our database for your specified metrics/key figures and respons with the data for all the selected stocks."
    verbose: Verbose keys
        camel: Camel Case
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/datatable/query/id={is_id}"
    querystring = {}
    if verbose:
        querystring['verbose'] = verbose
    if camel:
        querystring['camel'] = camel
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cashflow_statement_retrieve(identifier: str, index: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We serve 6 categories for 10-Q data. To combine all of them use the '10-Q' endpoint(s)."
    index: The index you would like to receive for a request of stock XY. Starts with 0 as latest quarter
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/cashflow/statement/{identifier}/"
    querystring = {}
    if index:
        querystring['index'] = index
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def financial_growth_list(page_size: int=None, index: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We serve 6 categories for 10-Q data. To combine all of them use the '10-Q' endpoint(s)."
    page_size: Number of results to return per page.
        index: The index you would like to receive for a request of stock XY. Starts with 0 as latest quarter
        page: A page number within the paginated result set.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/financial/growth/"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if index:
        querystring['index'] = index
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def financial_ratios_retrieve(identifier: str, index: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We serve 6 categories for 10-Q data. To combine all of them use the '10-Q' endpoint(s)."
    index: The index you would like to receive for a request of stock XY. Starts with 0 as latest quarter
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/financial/ratios/{identifier}/"
    querystring = {}
    if index:
        querystring['index'] = index
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def income_statement_list(page_size: int=None, index: str=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We serve 6 categories for 10-Q data. To combine all of them use the '10-Q' endpoint(s)."
    page_size: Number of results to return per page.
        index: The index you would like to receive for a request of stock XY. Starts with 0 as latest quarter
        page: A page number within the paginated result set.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/income/statement/"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if index:
        querystring['index'] = index
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def income_statement_retrieve(identifier: str, index: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We serve 6 categories for 10-Q data. To combine all of them use the '10-Q' endpoint(s)."
    index: The index you would like to receive for a request of stock XY. Starts with 0 as latest quarter
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/income/statement/{identifier}/"
    querystring = {}
    if index:
        querystring['index'] = index
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def longterm_analysis_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Long Term Scores are calculated based on the date of the next 10-Q Report. Check the docs for explanation"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/longterm/analysis/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cashflow_statement_list(page_size: int=None, page: int=None, index: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We serve 6 categories for 10-Q data. To combine all of them use the '10-Q' endpoint(s)."
    page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        index: The index you would like to receive for a request of stock XY. Starts with 0 as latest quarter
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/cashflow/statement/"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    if index:
        querystring['index'] = index
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keymetrics_retrieve(identifier: str, index: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We serve 6 categories for 10-Q data. To combine all of them use the '10-Q' endpoint(s)."
    index: The index you would like to receive for a request of stock XY. Starts with 0 as latest quarter
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/keymetrics/{identifier}/"
    querystring = {}
    if index:
        querystring['index'] = index
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def financial_growth_retrieve(identifier: str, index: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We serve 6 categories for 10-Q data. To combine all of them use the '10-Q' endpoint(s)."
    index: The index you would like to receive for a request of stock XY. Starts with 0 as latest quarter
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/financial/growth/{identifier}/"
    querystring = {}
    if index:
        querystring['index'] = index
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def financial_ratios_list(index: str=None, page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We serve 6 categories for 10-Q data. To combine all of them use the '10-Q' endpoint(s)."
    index: The index you would like to receive for a request of stock XY. Starts with 0 as latest quarter
        page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/financial/ratios/"
    querystring = {}
    if index:
        querystring['index'] = index
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def balance_sheet_retrieve(identifier: str, index: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We serve 6 categories for 10-Q data. To combine all of them use the '10-Q' endpoint(s)."
    index: The index you would like to receive for a request of stock XY. Starts with 0 as latest quarter
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/balance/sheet/{identifier}/"
    querystring = {}
    if index:
        querystring['index'] = index
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keymetrics_list(page_size: int=None, page: int=None, index: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We serve 6 categories for 10-Q data. To combine all of them use the '10-Q' endpoint(s)."
    page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        index: The index you would like to receive for a request of stock XY. Starts with 0 as latest quarter
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/keymetrics/"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    if index:
        querystring['index'] = index
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def balance_sheet_list(index: str=None, page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We serve 6 categories for 10-Q data. To combine all of them use the '10-Q' endpoint(s)."
    index: The index you would like to receive for a request of stock XY. Starts with 0 as latest quarter
        page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/balance/sheet/"
    querystring = {}
    if index:
        querystring['index'] = index
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def datatables_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Premium: Interact with your table by querying it with its unique ID. You can list, retrieve, delete, update, create and replace tables on the go."
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/datatables/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sectors_movers_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all sectors& their winners and losers (stocks)"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/sectors/movers/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shortterm_analysis_list(page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Following Week Score is calculated once every week. Check the docs for explanation"
    page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/shortterm/analysis/"
    querystring = {}
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def technicals_analysis_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Technical Scores are calculated multiple times a day. Check the docs for explanation"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/technicals/analysis/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sectors_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A detailed view on a specific sector. Find it by referring the Sector ID."
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/sectors/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def technicals_analysis_list(page_size: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Technical Scores are calculated multiple times a day. Check the docs for explanation"
    page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/technicals/analysis/"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def technicals_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive a stock with its Technical Indicators"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/technicals/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def technicals_list(page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive an up to list of stocks with their Technical Indicators"
    page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/technicals/"
    querystring = {}
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_10q_list(page_size: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Useful to fetch the latest quarterly dataset for a stock or multiple stocks."
    page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/10Q/"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sectors_list(page_size: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all sectors. Contains basic and time critical data."
    page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/sectors/"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sectors_movers_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A detailed view on a specific sector. Find it by referring the Sector ID."
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/sectors/movers/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_10q_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Useful to fetch the latest quarterly dataset for a stock or multiple stocks."
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/10Q/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quote_expand_stock_retrieve(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For premium: Receive an expanded quote with the underlying static, realtime and expanded information (Sector, Exchange)"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/quote/expand/stock={ticker}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quote_list(exchange: int=None, page_size: int=None, industry: str=None, ipo: str=None, page: int=None, country_short_name: str=None, is_id: int=None, ticker: str=None, company: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides stock quotes with time critical information, such as price and trading volume"
    page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/quote/"
    querystring = {}
    if exchange:
        querystring['exchange'] = exchange
    if page_size:
        querystring['page_size'] = page_size
    if industry:
        querystring['industry'] = industry
    if ipo:
        querystring['ipo'] = ipo
    if page:
        querystring['page'] = page
    if country_short_name:
        querystring['country__short_name'] = country_short_name
    if is_id:
        querystring['id'] = is_id
    if ticker:
        querystring['ticker'] = ticker
    if company:
        querystring['company'] = company
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tickers_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list response of all stock tickers with their ID, that are possible to trade"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/tickers/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quote_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You receive a detail stock quote with time critical data, f.ex. the price or market cap. of AAPL"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/quote/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sentiments_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single sentiment by its ID or the title."
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/sentiments/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shortterm_analysis_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Technical Scores are calculated multiple times a day. Check the docs for explanation"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/shortterm/analysis/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange_listed_tickers_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all stocks listed to your specific Exchange. Refer the exchange by its short name, f.ex. NYSE."
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/exchange/listed-tickers/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forex_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive an up to date list Forex pairs. They are updated nearly realtime"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/forex/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sectors_history_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all sectors& their historical performances (daily)"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/sectors/history/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reports_list(page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Premium: A list of all reports created by Palmy Investing"
    page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/reports/"
    querystring = {}
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forex_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query a specific Forex pair by ID."
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/forex/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sectors_history_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A detailed view on a specific sector. Find it by referring the Sector ID."
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/sectors/history/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A detailed view on a specific exchange. Find it by referring the Exchange ID or its short name (f.ex. NYSE)."
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/exchange/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all exchanges. Contains basic and time critical data."
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/exchange/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tickers_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list response of all stock tickers with their ID, that are possible to trade"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/tickers/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange_listed_tickers_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all stocks listed on all exchanges."
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/exchange/listed-tickers/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a news article by its id or title."
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/articles/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sentiments_list(page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Here we provide a list of all sentiments. A sentiment is computed via NLP"
    page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/sentiments/"
    querystring = {}
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reports_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Premium: Retrieve one of the reports by their ID"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/reports/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def articles_list(page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Here we provide a list of news articles."
    page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/articles/"
    querystring = {}
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def etf_assets_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all assets of your ETF. Search by symbol/ID."
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/etf/assets/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keyfigures_retrieve(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of all metrics you can assign for i.e data tables"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/keyfigures/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def etf_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all important price information of an ETF"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/etf/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def etf_assets_list(page_size: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of ETFs with all their assets (i.e. stocks, commodities)."
    page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/etf/assets/"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def etf_expand_list(page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Premium: A list of ETFs with expanded functionality. Here you can view Sectors& Countries."
    page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/etf/expand/"
    querystring = {}
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def etf_list(page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of n ETFs where each hold time critical information such as price"
    page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/etf/"
    querystring = {}
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def etf_expand_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Premium: Retrieve price and compotion information for an ETF. Search by ID/Symbol"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/etf/expand/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timeseries_stats_retrieve(identifier: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all important price statistics of an asset"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/timeseries/stats/{identifier}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timeseries_stats_list(page_size: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use lists full of price statistics to compare and analyse stocks"
    page_size: Number of results to return per page.
        page: A page number within the paginated result set.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/timeseries/stats/"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def timeseries_id_retrieve(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all important daily price information about a stock"
    
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/timeseries/id={is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def longterm_analysis_list(page: int=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Long Term Scores are calculated based on the date of the next 10-Q Report. Check the docs for explanation"
    page: A page number within the paginated result set.
        page_size: Number of results to return per page.
        
    """
    url = f"https://palmy-investing-api.p.rapidapi.com/api/longterm/analysis/"
    querystring = {}
    if page:
        querystring['page'] = page
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palmy-investing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

