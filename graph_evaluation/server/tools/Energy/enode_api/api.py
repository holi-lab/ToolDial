import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def listhvacs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://enode-api.p.rapidapi.com/hvacs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gethealthchargervendors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists the available charger vendors. If you authenticate with a client or link token we also show the vendors that your client has beta access to."
    
    """
    url = f"https://enode-api.p.rapidapi.com/health/chargers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvehiclesaction(actionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the current state of the requested action."
    actionid: ID of the Action
        
    """
    url = f"https://enode-api.p.rapidapi.com/vehicles/actions/{actionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvehiclesvehicleidsmartchargingpolicy(vehicleid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a vehicle's [Smart Charging](/docs/smart-charging/introduction) policy"
    vehicleid: ID of the Vehicle.
        
    """
    url = f"https://enode-api.p.rapidapi.com/vehicles/{vehicleid}/smart-charging-policy"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstatisticsproduction(startdate: str, type: str='<string>', charginglocationid: str='<string>', utcoffset: int=-8, enddate: str='<date>', resolution: str='DAY', is_id: str='<string>', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a normalized time series of statistics about power production and price for the User."
    startdate: Earliest date to include in the response. Cannot be greater than endDate.
        type: Get statistics for this hardware type.
        charginglocationid: Filter statistics to only include this charging location.
        utcoffset: Offset (in hours) from UTC of the timezone from which the statistics should be viewed. By default, all returned timestamps are in UTC, and period boundaries (day, week, month, year) used in the aggregation are calculated in UTC. Providing `utcOffset` instead aligns these to the viewer's timezone so that the timestamps and period boundaries fall where the viewer expects them to. Positive, negative, and fractional values are valid.
        enddate: Latest date to include in the response (defaults to current date/time).
        resolution: The unit of time the data will be cut into before aggregate statistics are applied. Each entry in the response array corresponds to aggregated data of the time range specified.
        is_id: Filter statistics to only include a specific entity. Hardware category of the entity must match the `type` parameter.
        
    """
    url = f"https://enode-api.p.rapidapi.com/statistics/production"
    querystring = {'startDate': startdate, }
    if type:
        querystring['type'] = type
    if charginglocationid:
        querystring['chargingLocationId'] = charginglocationid
    if utcoffset:
        querystring['utcOffset'] = utcoffset
    if enddate:
        querystring['endDate'] = enddate
    if resolution:
        querystring['resolution'] = resolution
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcharginglocations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of Charging Locations registered to the User"
    
    """
    url = f"https://enode-api.p.rapidapi.com/charging-locations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gethvac(hvacid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    hvacid: ID of the HVAC
        
    """
    url = f"https://enode-api.p.rapidapi.com/hvacs/{hvacid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gethvacsaction(actionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the current state of the requested action."
    actionid: ID of the Action
        
    """
    url = f"https://enode-api.p.rapidapi.com/hvacs/actions/{actionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gethealthinvertervendors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists the available inverter vendors. If you authenticate with a client or link token we also show the vendors that your client has beta access to."
    
    """
    url = f"https://enode-api.p.rapidapi.com/health/inverters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gethealthhvacvendors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists the available HVAC vendors. If you authenticate with a client token we also show the vendors that your client has beta access to."
    
    """
    url = f"https://enode-api.p.rapidapi.com/health/hvacs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcharginglocationscharginglocationid(charginglocationid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    charginglocationid: ID of the Charging Location.
        
    """
    url = f"https://enode-api.p.rapidapi.com/charging-locations/{charginglocationid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gethealthvehiclevendors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists the available vehicle vendors. If you authenticate with a client or link token we also include the vendors that your client has beta access to."
    
    """
    url = f"https://enode-api.p.rapidapi.com/health/vehicles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvehiclesvehicleidsmartchargingstatus(vehicleid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Every vehicle in Enode has a [SmartChargingStatus](/docs/smart-charging/introduction) object that describes the vehicle in terms of smart charging."
    vehicleid: ID of the Vehicle.
        
    """
    url = f"https://enode-api.p.rapidapi.com/vehicles/{vehicleid}/smart-charging-status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcharginglocationtariff(charginglocationid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Tariff intervals for a given charging location."
    charginglocationid: ID of the Charging Location
        
    """
    url = f"https://enode-api.p.rapidapi.com/charging-locations/{charginglocationid}/tariff"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getschedule(scheduleid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    scheduleid: ID of the Schedule.
        
    """
    url = f"https://enode-api.p.rapidapi.com/schedules/{scheduleid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getschedulestatus(scheduleid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    scheduleid: ID of the Schedule.
        
    """
    url = f"https://enode-api.p.rapidapi.com/schedules/{scheduleid}/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getme(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns metadata about the authenticated User, including a list of vendors for which the User has provided credentials."
    
    """
    url = f"https://enode-api.p.rapidapi.com/me"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getstatisticscharging(startdate: str, enddate: str='<date>', charginglocationid: str='<string>', resolution: str='DAY', utcoffset: int=-8, is_id: str='<string>', type: str='<string>', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a normalized time series of statistics about power consumption and price for the User.
		
		If Smart Charging has shifted the consumption, the 'non-smart' price fields will show what the consumption would have cost if it had happened at the default time. The difference between the two is provided by the `estimatedSavings` field for convenience. `<CURRENCY>` is an ISO4217 Alpha-3 currency code that is determined by client-wide configuration or the currency code provided during price data ingestion (such as Tariffs)."
    startdate: Earliest date to include in the response. Cannot be greater than endDate.
        enddate: Latest date to include in the response (defaults to current date/time).
        charginglocationid: Filter statistics to only include this charging location.
        resolution: The unit of time the data will be cut into before aggregate statistics are applied. Each entry in the response array corresponds to aggregated data of the time range specified.
        utcoffset: Offset (in hours) from UTC of the timezone from which the statistics should be viewed. By default, all returned timestamps are in UTC, and period boundaries (day, week, month, year) used in the aggregation are calculated in UTC. Providing `utcOffset` instead aligns these to the viewer's timezone so that the timestamps and period boundaries fall where the viewer expects them to. Positive, negative, and fractional values are valid.
        is_id: Filter statistics to only include a specific entity. Hardware category of the entity must match the `type` parameter.
        type: Get statistics for this hardware type.
        
    """
    url = f"https://enode-api.p.rapidapi.com/statistics/charging"
    querystring = {'startDate': startdate, }
    if enddate:
        querystring['endDate'] = enddate
    if charginglocationid:
        querystring['chargingLocationId'] = charginglocationid
    if resolution:
        querystring['resolution'] = resolution
    if utcoffset:
        querystring['utcOffset'] = utcoffset
    if is_id:
        querystring['id'] = is_id
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listinverters(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://enode-api.p.rapidapi.com/inverters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvehiclesvehicleidsmartchargingplans(vehicleid: str, smartchargingplanid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check status of current or historical Smart Charging Plan for a Vehicle. Returns a [Smart Charging](/docs/smart-charging/introduction) Plan for this vehicle.
		
		To fetch the most recently created plan, call the endpoint with `smartChargingPlanId` set to `latest`."
    vehicleid: ID of the Vehicle.
        smartchargingplanid: ID of the Smart Charging Plan
        
    """
    url = f"https://enode-api.p.rapidapi.com/vehicles/{vehicleid}/smart-charging-plans/{smartchargingplanid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getchargersaction(actionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the current state of the requested action."
    actionid: ID of the Action
        
    """
    url = f"https://enode-api.p.rapidapi.com/chargers/actions/{actionid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getschedules(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of Schedules registered to the User."
    
    """
    url = f"https://enode-api.p.rapidapi.com/schedules"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listchargers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available Chargers for the User."
    
    """
    url = f"https://enode-api.p.rapidapi.com/chargers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "enode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

