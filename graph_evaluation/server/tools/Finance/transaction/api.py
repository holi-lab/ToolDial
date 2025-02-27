import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getpaymentinitiationinstructionsummary(limit: int=25, end_to_end_identification: str='ABC/ABC-13679/2021-01-20', offset: int=0, creation_to_date: str='2022-05-31', requested_execution_date: str='2021-09-15', instructed_amount: str='1000000.00', creation_from_date: str='2022-05-30', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all payment initiation instructions allows to retrieve all customer credit transfer initiations. If the number of results exceeds the default limit, the response will be paginated. The operation allows filtering the results based on the creation from date and creation to date. The result is filtered based on "creation_date_time". The endpoint will return an empty array when no resource is found.
		"
    limit: The maximum number of items to return in a page. If unspecified, the default limit will be returned. If the limit specified is greater than the maximum permitted by the API, the API will return the maximum permitted limit. Each API should define *maximum and *default limit.
        end_to_end_identification: Unique identification, as assigned by the initiating party, to unambiguously identify the transaction. The consumer of the API can filter the response using end to end identification when the "uetr" is not available.
        offset: Specifies the offset / starting point in the list of all available items, starting from which results will be returned. The numeric offset identifies the page token, allowing users to advance to the next page in the collection. The value 0 (zero) identifies the first page of entry.
        creation_to_date: The query parameter can use either a specific creation-from-date or date range in the form of creation-from-date and creation-to-date. For a specific date creation-to-date must not be provided. The result will be filtered on creation_date_time. If a origin server do not support the time component, it can ignore the time component. Expressed in YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS format. For example - 2021-04-01T12:01:01.000 or 2021-04-01T12:01:01
        requested_execution_date: Date and time at which the payment is executed and the cash is at the disposal of the credit account owner. The result will be filtered on requested_execution_date. If a origin server do not support the time component, it can ignore the time component. Expressed in YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS format. For example - 2021-04-01T12:01:01.000 or 2021-04-01T12:01:01
        instructed_amount: Specifies the amount as ordered in the payment initiation instruction before any deduction.

        creation_from_date: The query parameter can use either a specific creation-from-date or date range in the form of creation-from-date and creation-to-date. The result will be filtered on creation_date_time. If a origin server do not support the time component, it can ignore the time component. Expressed in YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS format. For example - 2021-04-01T12:01:01.000 or 2021-04-01T12:01:01
        
    """
    url = f"https://transaction5.p.rapidapi.com/payment-initiation-instructions"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if end_to_end_identification:
        querystring['end-to-end-identification'] = end_to_end_identification
    if offset:
        querystring['offset'] = offset
    if creation_to_date:
        querystring['creation-to-date'] = creation_to_date
    if requested_execution_date:
        querystring['requested-execution-date'] = requested_execution_date
    if instructed_amount:
        querystring['instructed-amount'] = instructed_amount
    if creation_from_date:
        querystring['creation-from-date'] = creation_from_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transaction5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdirectdebitinitiationinstructionsummary(creation_from_date: str='2022-05-30', offset: int=0, creation_to_date: str='2022-05-31', end_to_end_identification: str='ABC/ABC-13679/2021-01-20', limit: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all direct debit initiation instructions allows to retrieve all direct debit initiation instruction. If the number of results exceeds the default limit, the response will be paginated. The operation allows filtering the results based on the creation from date and creation to date. The result is filtered based on "creation_date_time". The endpoint will return an empty array when no resource is found.
		"
    creation_from_date: The query parameter can use either a specific creation-from-date or date range in the form of creation-from-date and creation-to-date. The result will be filtered on creation_date_time. If a origin server do not support the time component, it can ignore the time component. Expressed in YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS format. For example - 2021-04-01T12:01:01.000 or 2021-04-01T12:01:01
        offset: Specifies the offset / starting point in the list of all available items, starting from which results will be returned. The numeric offset identifies the page token, allowing users to advance to the next page in the collection. The value 0 (zero) identifies the first page of entry.
        creation_to_date: The query parameter can use either a specific creation-from-date or date range in the form of creation-from-date and creation-to-date. For a specific date creation-to-date must not be provided. The result will be filtered on creation_date_time. If a origin server do not support the time component, it can ignore the time component. Expressed in YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS format. For example - 2021-04-01T12:01:01.000 or 2021-04-01T12:01:01
        end_to_end_identification: Unique identification, as assigned by the initiating party, to unambiguously identify the transaction. The consumer of the API can filter the response using end to end identification when the "uetr" is not available.
        limit: The maximum number of items to return in a page. If unspecified, the default limit will be returned. If the limit specified is greater than the maximum permitted by the API, the API will return the maximum permitted limit. Each API should define *maximum and *default limit.
        
    """
    url = f"https://transaction5.p.rapidapi.com/direct-debit-initiation-instruction"
    querystring = {}
    if creation_from_date:
        querystring['creation-from-date'] = creation_from_date
    if offset:
        querystring['offset'] = offset
    if creation_to_date:
        querystring['creation-to-date'] = creation_to_date
    if end_to_end_identification:
        querystring['end-to-end-identification'] = end_to_end_identification
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transaction5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdirectdebitinitiationinstruction(uetr: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this operation to retrieve a direct debit initiation instruction by ID (UETR)."
    uetr: An RFC4122 UID used as a unique Payment Instruction Identifier.
        
    """
    url = f"https://transaction5.p.rapidapi.com/direct-debit-initiation-instruction/{uetr}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transaction5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpaymentinitiationinstruction(uetr: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this operation to retrieve a payment initiation instruction by ID (UETR)."
    uetr: An RFC4122 UID used as a unique Payment Instruction Identifier.
        
    """
    url = f"https://transaction5.p.rapidapi.com/payment-initiation-instructions/{uetr}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "transaction5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

