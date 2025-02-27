import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_sec_administrative_proceedings(fromdate: str='2019', todate: str='2020', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoints provides data relating to notices and orders concerning the institution and/or settlement of SEC administrative proceedings.
		
		Current and Historical Data."
    fromdate: Effective Year.

Default value is current year.
        todate: Effective Year.

Default value is current year.
        
    """
    url = f"https://sec-enforcement-actions.p.rapidapi.com/GetAdminProceedings"
    querystring = {}
    if fromdate:
        querystring['fromDate'] = fromdate
    if todate:
        querystring['toDate'] = todate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-enforcement-actions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sec_delinquent_filings(todate: str='2020', fromdate: str='2019', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "These endpoint displays SEC enforcement actions, orders, and opinions issued by the Commission in administrative proceedings relating to delinquent filings.
		
		Current and Historical Data."
    todate: Effective Year. 

Default is Current Year.
        fromdate: Effective Year. 

Default is Current Year.
        
    """
    url = f"https://sec-enforcement-actions.p.rapidapi.com/GetDelinquentFilings"
    querystring = {}
    if todate:
        querystring['toDate'] = todate
    if fromdate:
        querystring['fromDate'] = fromdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-enforcement-actions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sec_trading_suspensions(fromdate: str='2019', todate: str='2020', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint displays list of companies with recent SEC trading suspensions. 
		
		SEC is able to suspend trading in any stock for up to ten trading days when the SEC determines that a trading suspension is required in the public interest and for the protection of investors.
		
		Current and Historical Data."
    fromdate: Effective Date.

Default value is current year.
        todate: Effective Date.

Default value is current year.
        
    """
    url = f"https://sec-enforcement-actions.p.rapidapi.com/GetTradingSuspensions"
    querystring = {}
    if fromdate:
        querystring['fromDate'] = fromdate
    if todate:
        querystring['toDate'] = todate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-enforcement-actions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sec_litigations(todate: str='2020', fromdate: str='2019', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides data related to litigation releases concerning civil lawsuits brought by the SEC Commission in federal court.
		
		Current and Historical Data."
    todate: Effective Year.

Default value is current year.
        fromdate: Effective Year.

Default value is current year.
        
    """
    url = f"https://sec-enforcement-actions.p.rapidapi.com/GetLitigations"
    querystring = {}
    if todate:
        querystring['toDate'] = todate
    if fromdate:
        querystring['fromDate'] = fromdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-enforcement-actions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

