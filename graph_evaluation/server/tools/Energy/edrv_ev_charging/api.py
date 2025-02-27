import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getchargestationconnectors(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Connectors data filtered by ChargeStation"
    id: The ChargeStation id that connectors need to be fetched for
        
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/chargestations/{is_id}/connectors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getchargestation(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use to request a single charge station's data"
    id: The charge station id that needs to be fetched
        
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/chargestations/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconnector(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one Connector data"
    id: ID of Connector that needs to be fetched
        
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/connectors/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrealtime(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use to request a Websockets handshake"
    
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/realtime"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getreservations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Reservations data"
    
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/reservations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getchargestations(organization: str=None, location: str=None, active: bool=None, online: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ChargeStation info"
    organization: Filter by Org. Id
        location: Filter by Location Id
        active: Chargestations that have been activated/deactivated by the admin
        online: Filter by Online Status
        
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/chargestations"
    querystring = {}
    if organization:
        querystring['organization'] = organization
    if location:
        querystring['location'] = location
    if active:
        querystring['active'] = active
    if online:
        querystring['online'] = online
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getorganization(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one organization data by id"
    id: ID of organization that needs to be fetched
        
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/organizations/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstat(asset: str, statistic: str, to: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get statistics for an activity for an asset within a certain date"
    asset: The asset you would like to get stats for.
        statistic: Type of statistic to calculate
        to: Date as ISO String
        from: Date as ISO String
        
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/stats/{asset}/{statistic}"
    querystring = {'to': to, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdriver(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use to request a single driver"
    id: The driver id that needs to be fetched
        
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/drivers/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettransaction(is_id: str, driver: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific transaction"
    id: BSON
        driver: Return Driver Info based on Identification Method used for this transaction
        
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/transactions/{is_id}"
    querystring = {}
    if driver:
        querystring['driver'] = driver
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconfigurations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Configurations data"
    
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/configurations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getorganizations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Organizations data"
    
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/organizations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getreservation(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one reservation data"
    id: ID of the reservation that needs to be fetched
        
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/reservations/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconnectors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Connectors data"
    
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/connectors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettransactions(to: str, is_from: str, onlyactive: bool=None, page: str=None, sortby: str=None, limit: int=10, sortascending: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all transactions"
    to: Date as ISO String
        from: Date as ISO String
        onlyactive: True to get only active transactions
        page: The queried page index.
        sortby: Sort data by this key
        limit: Number of results per page
        sortascending: True to sort ascending (default is false - descending)
        
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/transactions"
    querystring = {'to': to, 'from': is_from, }
    if onlyactive:
        querystring['onlyActive'] = onlyactive
    if page:
        querystring['page'] = page
    if sortby:
        querystring['sortBy'] = sortby
    if limit:
        querystring['limit'] = limit
    if sortascending:
        querystring['sortAscending'] = sortascending
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdrivers(active: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get All Drivers data"
    active: Get a list of active drivers
        
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/drivers"
    querystring = {}
    if active:
        querystring['active'] = active
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconfiguration(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one Configuration data"
    id: ID of Configuration that needs to be fetched
        
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/configurations/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettoken(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use to request a single token"
    id: The token id that needs to be fetched
        
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/tokens/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcommands(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Commands data"
    
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/commands"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettokens(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Tokens data"
    
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/tokens"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getlocations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Locations data"
    
    """
    url = f"https://edrv-ev-charging.p.rapidapi.com/v1/locations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edrv-ev-charging.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

