import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def index2(content_type: str, request_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameter | Default | Description
		--------- | ------- | -----------
		start_date | (conditional) | Start date. UTC format. Mutually exclusive with request_id and status.
		end_date | (conditional) | End date. UTC format. Mutually exclusive with request_id and status.
		request_id | (conditional) | Request ID (used when creating the label). Mutually exclusive with date range and status.
		status | (conditional) | Label status. Mutually exclusive with date range and request_id.
		page_size | 25 | Page size.
		page_number | 1 | Page number."
    
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/manifests"
    querystring = {'Content-Type': content_type, 'request_id': request_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstreets(content_type: str, zip5: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameter | Default | Description
		--------- | ------- | -----------
		zip5 | (required) | Zip 5."
    
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/address/streets"
    querystring = {'Content-Type': content_type, 'zip5': zip5, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def index(content_type: str, request_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameter | Default | Description
		--------- | ------- | -----------
		start_date | (conditional) | Start date. UTC format. Mutually exclusive with request_id and status.
		end_date | (conditional) | End date. UTC format. Mutually exclusive with request_id and status.
		request_id | (conditional) | Request ID (used when creating the label). Mutually exclusive with date range and status.
		status | (conditional) | Label status. Mutually exclusive with date range and request_id.
		page_size | 25 | Page size.
		page_number | 1 | Page number."
    
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/labels"
    querystring = {'Content-Type': content_type, 'request_id': request_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reprint1(content_type: str, manifest_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameter | Default | Description
		--------- | ------- | -----------
		manifest_number | (required) | Manifest number."
    
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/manifests/{manifest_number}"
    querystring = {'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auth(content_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/authenticate"
    querystring = {'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcity(content_type: str, zip5: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameter | Default | Description
		--------- | ------- | -----------
		zip5 | (required) | Zip 5."
    
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/address/city"
    querystring = {'Content-Type': content_type, 'zip5': zip5, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def read(content_type: str, authorization: str, user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/users/{user_id}"
    querystring = {'Content-Type': content_type, 'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpossibleaddresses(content_type: str, primary: int, street_name: str, zip5: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameter | Default | Description
		--------- | ------- | -----------
		zip5 | (required) | Zip 5.
		primary | (required) | Primary street number.
		street_name | (null/empty) | Street name. Could be null, incomplete (e.g. "H") or full (e.g. "High")"
    
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/address/possible_addresses"
    querystring = {'Content-Type': content_type, 'primary': primary, 'street_name': street_name, 'zip5': zip5, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reprint(authorization: str, tracking_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<aside class="notice">
		  Not available for UPS.
		</aside>
		
		Parameter | Default | Description
		--------- | ------- | -----------
		tracking_number | (required) | Tracking number."
    
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/labels/{tracking_number}"
    querystring = {'Authorization': authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def track(tracking_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameter | Default | Description
		--------- | ------- | -----------
		tracking_number | (required) | Tracking number."
    
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/track/{tracking_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def index0(content_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Context: user
		
		Parameter | Default | Description
		--------- | ------- | -----------
		page_size | 25 | Page size.
		page_number | 1 | Page number."
    
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/payment_methods"
    querystring = {'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def account_ledger(entry_type: str, start_date: str, end_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameter | Default | Description
		--------- | ------- | -----------
		entry_type | (null/empty) | Could have one of the 4 possible values: "label_charge", "adjustment","money_deposit", "refund" "
    
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/account_ledger"
    querystring = {'entry_type': entry_type, 'start_date': start_date, 'end_date': end_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pickups(page_number: str, page_size: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/pickups"
    querystring = {'page_number': page_number, 'page_size': page_size, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pickup(pickup_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/pickups/{pickup_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def refunds(start_date: str, end_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameter | Default | Description
		--------- | ------- | -----------
		start_date | (required) | Start date of date range when refunded labels are created
		end_date | (required) | End date of date range when refunded labels are created"
    start_date: start date of created_at label date range
        end_date: end date of created_at label date range
        
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/refunds"
    querystring = {'start_date': start_date, 'end_date': end_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def index0(content_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Parameter | Default | Description
		--------- | ------- | -----------
		page_size | 25 | Page size.
		page_number | 1 | Page number."
    
    """
    url = f"https://international-bridge-blue.p.rapidapi.com/v1/payment_transactions"
    querystring = {'Content-Type': content_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "international-bridge-blue.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

