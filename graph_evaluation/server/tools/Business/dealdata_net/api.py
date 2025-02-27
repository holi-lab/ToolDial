import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_industries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This returns all of the available industries, or company types, in the database."
    
    """
    url = f"https://fintoolbox-dealdata-net-v1.p.rapidapi.com/api.php?function=getIndustries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fintoolbox-dealdata-net-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filing_profile_from_filing_id(filingid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given an internal filing ID (returned from another endpoint), this call returns a full filing profile."
    
    """
    url = f"https://fintoolbox-dealdata-net-v1.p.rapidapi.com/api.php?function=filingProfile&filingid={filingid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fintoolbox-dealdata-net-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def broker_search(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of brokers that match the given name fragment. NOTE: API responses are limited to 50,000 characters so the fragment provided must limit the result set so as not to exceed this limit."
    
    """
    url = f"https://fintoolbox-dealdata-net-v1.p.rapidapi.com/api.php?function=findBroker&name={name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fintoolbox-dealdata-net-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def person_profile_from_person_id(personid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given the person ID from a person search API call, this endpoint returns a full profile for that person."
    
    """
    url = f"https://fintoolbox-dealdata-net-v1.p.rapidapi.com/api.php?function=personProfile&personid={personid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fintoolbox-dealdata-net-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fund_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of available fund types for FUND filing feeds and searches."
    
    """
    url = f"https://fintoolbox-dealdata-net-v1.p.rapidapi.com/api.php?function=getFundTypes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fintoolbox-dealdata-net-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def person_search(first: str=None, middle: str=None, last: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fragments can be of any length, including a null string. NOTE: API call returns are limited to more than 50,000 characters each. As a result, you should provide enough of a guess as to the peroson's name so as to adequately limit the result set."
    first: One or more characters from the person's first name.
        middle: One or more characters from the person's middle name.
        last: One or more characters from the person's last name.
        
    """
    url = f"https://fintoolbox-dealdata-net-v1.p.rapidapi.com/api.php"
    querystring = {}
    if first:
        querystring['first'] = first
    if middle:
        querystring['middle'] = middle
    if last:
        querystring['last'] = last
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fintoolbox-dealdata-net-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filing_feed_search(entity: str, limit: int, first: int, start: str, type: str=None, amountlow: int=None, amounthigh: int=None, state: str=None, end: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns a filing feed based on any combination of a date range, entity type, range of amount raised, primary company/fund state.  It also allows a limit range to be set.  NOTE: all JSON return strings are limited to a maximum of 50 kilobytes.  Ensure that your query is sufficiently narrow, or an explicit result limit (<= 25 per call recommended) is set to ensure that the call does not return a 0 status."
    entity: May be "FUND", "COMPANY", or "ALL".  Only filings of the specified type will be returned.
        limit: The maximum number of results to return.  It is not recommended to set this value higher than 25.  NOTE: All JSON values are limited to 50 kilobytes in size. If you set this parameter too large, your result set will not be returned.
        first: The first result to return (starting from 0).  Modify this value to page through large result sets.
        start: A start date of the format YYYY-mm-dd .  Only filings since this date (inclusive) will be returned.  Since filings are returned in descending date order, this parameter can be set to the distant past to not limit by date.
        type: May be set to one of the values returned by the Discover Available Parameters endpoints.  Separate values are provided for funds and companies.
        amountlow: Only filings where the total amount sold is greater than this parameter will be returned.  Leave blank for no lower bound.
        amounthigh: Only filings where no more than amounthigh USD was raised will be returned.  Leave blank for no upper bound.
        state: Only filings whose primary entity is located in this two-letter US state code will be returned.  Leave blank for all states.
        end: Optional end date (of the form YYYY-mm-dd) for the result set. If this is not set, the current date will be used. as the end date.
        
    """
    url = f"https://fintoolbox-dealdata-net-v1.p.rapidapi.com/api.php"
    querystring = {'entity': entity, 'limit': limit, 'first': first, 'start': start, }
    if type:
        querystring['type'] = type
    if amountlow:
        querystring['amountlow'] = amountlow
    if amounthigh:
        querystring['amounthigh'] = amounthigh
    if state:
        querystring['state'] = state
    if end:
        querystring['end'] = end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fintoolbox-dealdata-net-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def broker_profile_from_broker_id(brokerid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given the broker ID returned from a broker search, this endpoint returns a full broker profile."
    
    """
    url = f"https://fintoolbox-dealdata-net-v1.p.rapidapi.com/api.php?function=brokerProfile&brokerid={brokerid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fintoolbox-dealdata-net-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_profile_from_cik(cik: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a CIK number (the SEC identifier for the entity), this function returns a full company or fund profile for the entity."
    
    """
    url = f"https://fintoolbox-dealdata-net-v1.p.rapidapi.com/api.php?function=companyProfile&cik={cik}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fintoolbox-dealdata-net-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def company_fund_search(name_fragment: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Finds all companies matching the name fragment. NOTE: All API requests are limited to a maximum return size of 50,000 characters. While there are no hard limits on how long of a fragment needs to be provided, the fragment provided should not return too many results or the response limit will be triggered."
    
    """
    url = f"https://fintoolbox-dealdata-net-v1.p.rapidapi.com/api.php?function=findCompany&name={name_fragment}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fintoolbox-dealdata-net-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def broker_profile_from_crd_number(crd_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a FINRA Central Registration Depository (CRD) for a broker/dealer, this function returns a full broker profile."
    
    """
    url = f"https://fintoolbox-dealdata-net-v1.p.rapidapi.com/api.php?function=brokerProfileCRD&crd={crd_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fintoolbox-dealdata-net-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

