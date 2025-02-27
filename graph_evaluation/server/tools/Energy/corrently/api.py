import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def omreadings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a list of available meterrs in the OpenMETER project ( https://www.openmeter.de/ ) which grants access for analytics as data discovery.
		"
    
    """
    url = f"https://corrently.p.rapidapi.com/alternative/openmeter/readings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gsidispatch(key: str=None, zip: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Dispatch of green energy has two aspects to consider:
		  - Availability of gerneration facility (depends on weather and installed capacity)
		  - Demand of energy
		Using the green power index (GrünstromIndex) we have received a tool to automate distribution of energy in order to prevent redispatch situations. Doing this alows to opimize resource usage (tactical) and leverage data for investment planning (strategic).
		"
    key: Any valid Stromkonto account (address).
        zip: Zipcode (Postleitzahl) of a city in Germany.
        
    """
    url = f"https://corrently.p.rapidapi.com/gsi/dispatch"
    querystring = {}
    if key:
        querystring['key'] = key
    if zip:
        querystring['zip'] = zip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ocppsessions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns lastSession info of OCCP Cloud service for clearing in corrently ecosystem. Might be tested via [OCPP cloud simulator](https://ocpp.corrently.cloud).
		Last session Info of managed EV charging stations connected to the correnty ecosystem.
		"
    
    """
    url = f"https://corrently.p.rapidapi.com/alternative/ocpp/lastSessions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def meteringget(account: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a metered reading using account (Stromkonto).
		"
    account: Account/Address (Stromkonto) to retrieve reading for.
        
    """
    url = f"https://corrently.p.rapidapi.com/metering/reading"
    querystring = {}
    if account:
        querystring['account'] = account
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tariffslph0(zipcode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides pricing data for private households with standard load profiles (Standardlastprofil H0).
		"
    zipcode: Zipcode (Postzleitzahl) of a city in Germany.
        
    """
    url = f"https://corrently.p.rapidapi.com/tariff/slph0"
    querystring = {}
    if zipcode:
        querystring['zipcode'] = zipcode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def co2compensate(kwh: int=30, liter: int=30, co2: int=12345, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You might call the /price API endpoint prior to a checkout to get latest pricing information. This method returns a URL for direct checkout using Stripe. All CO2 offset prices are based on VCS Verified CO2 Emission Reduction In accordance with the requirements of ISO 14064-1. CO2 prices are actual market prices and may change every hour.
		"
    kwh: Kilo Watt hours of electricity to co2 offset.
        liter: Liter of fuel to compensate.
        co2: CO2 equivalence in gram to compensate
        
    """
    url = f"https://corrently.p.rapidapi.com/co2/compensate"
    querystring = {}
    if kwh:
        querystring['kwh'] = kwh
    if liter:
        querystring['liter'] = liter
    if co2:
        querystring['co2'] = co2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stromkontobalances(account: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stromkonto represents a core component of the Corrently Ecosystem. It is a ledger for green energy related transactions and gets heavily used by the public Web-UI on www.stromkonto.net . Beside of some decoration and reformating operations all data is backed by the [Energychain blockchain](https://github.com/energychain/) to provide consensus of balances and transactions. Use this API Endppoint if you prefere not to work with low level Distributed Ledger Technology (Blockchain).
		"
    account: Ethereum style address referencing a valid account (AKA Stromkonto).
        
    """
    url = f"https://corrently.p.rapidapi.com/stromkonto/balances"
    querystring = {}
    if account:
        querystring['account'] = account
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gsibesthour(zip: str=None, timeframe: int=None, key: str=None, hours: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simple Wrapper around the GreenPowerIndex for easy integration into almost any SmartHome system that allows access to a JSON/REST Service This endpoint is designed to indicate if a device should be turned on or off. (Switch state).
		"
    zip: Zipcode (Postleitzahl) of a city in Germany.
        timeframe: Number of hours to check (default 24 hours from now).
        key: Any valid Stromkonto account (address).
        hours: How many hours in row do you need the device turned on?
        
    """
    url = f"https://corrently.p.rapidapi.com/gsi/bestHour"
    querystring = {}
    if zip:
        querystring['zip'] = zip
    if timeframe:
        querystring['timeframe'] = timeframe
    if key:
        querystring['key'] = key
    if hours:
        querystring['hours'] = hours
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gsimarketdata(zip: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Compatible to awattar (https://api.awattar.de/v1/marketdata) API interface but data comes from GreenPowerIndex instead of EPEXSpot.
		"
    zip: Zipcode (Postleitzahl) of a city in Germany.
        
    """
    url = f"https://corrently.p.rapidapi.com/gsi/marketdata"
    querystring = {}
    if zip:
        querystring['zip'] = zip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quittungzugferd(account: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows to retrieve XML of the zugferd invoice.
		"
    account: Quittung Identifier  (serialnumber generated during receipt generation process)
        
    """
    url = f"https://corrently.p.rapidapi.com/quittung/zugferd"
    querystring = {}
    if account:
        querystring['account'] = account
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def omactivities(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a list of available meterrs in the OpenMETER project ( https://www.openmeter.de/ ) which grants access for analytics as data discovery.
		"
    
    """
    url = f"https://corrently.p.rapidapi.com/alternative/openmeter/activities"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def easeesessions(username: str=None, password: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Refer to easee.cloud API for details.
		"
    username: Username as used on easy.cloud
        password: Password as used on easy.cloud
        
    """
    url = f"https://corrently.p.rapidapi.com/alternative/easee/lastSessions"
    querystring = {}
    if username:
        querystring['username'] = username
    if password:
        querystring['password'] = password
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wimstatus(vid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Access to status information of an existing metering change and allocation process.
		"
    vid: VID key of the process.
        
    """
    url = f"https://corrently.p.rapidapi.com/wim/status"
    querystring = {}
    if vid:
        querystring['vid'] = vid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def co2sources(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Valid and certified sources of co2 compensation.
		"
    
    """
    url = f"https://corrently.p.rapidapi.com/co2/sources"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stromkontochoices(account: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Signable choices (contract changes) for customer.
		"
    account: Ethereum style address referencing a valid account alias (never use Stromkonto directly!).
        
    """
    url = f"https://corrently.p.rapidapi.com/stromkonto/choices"
    querystring = {}
    if account:
        querystring['account'] = account
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def co2price(kwh: int=30, co2: int=12345, liter: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All CO2 offset prices are based on VCS Verified CO2 Emission Reduction In accordance with the requirements of ISO 14064-1. CO2 prices are actual market prices and may change every hour.
		"
    kwh: Kilo Watt hours of electricity to co2 offset.
        co2: CO2 equivalence in gram to compensate
        liter: Liter of fuel to compensate.
        
    """
    url = f"https://corrently.p.rapidapi.com/co2/price"
    querystring = {}
    if kwh:
        querystring['kwh'] = kwh
    if co2:
        querystring['co2'] = co2
    if liter:
        querystring['liter'] = liter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tariffcomponents(email: str=None, wh: int=None, zipcode: str=None, milliseconds: int=None, kwha: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides insides into the different cost components of energy for a private household.
		Sample Request: https://api.corrently.io/v2.0/tariff/components?email=demo%40corrently.io&zip=69168&kwha=3300
		"
    email: Valid email address to assign request to (pre offer generation). Ensure GDPR (DSGVO) at any time
        wh: If provided together with milliseconds, a cost component stament for a particular event (like charging a car) will be created.
        zipcode: Zipcode (Postzleitzahl) of a city in Germany.
        milliseconds: If provided all results will be scaled to this timeframe
        kwha: Total amount of energy in kilo-watt-hours per year. (sample 2100)
        
    """
    url = f"https://corrently.p.rapidapi.com/tariff/components"
    querystring = {}
    if email:
        querystring['email'] = email
    if wh:
        querystring['wh'] = wh
    if zipcode:
        querystring['zipcode'] = zipcode
    if milliseconds:
        querystring['milliseconds'] = milliseconds
    if kwha:
        querystring['kwha'] = kwha
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gsiprediction(zip: str=None, key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieval the GreenPowerIndex (GrünstromIndex) for a given city (by zipcode) in Germany.
		"
    zip: Zipcode (Postleitzahl) of a city in Germany.
        key: Any valid Stromkonto account (address).
        
    """
    url = f"https://corrently.p.rapidapi.com/gsi/prediction"
    querystring = {}
    if zip:
        querystring['zip'] = zip
    if key:
        querystring['key'] = key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ommeters(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a list of available meterrs in the OpenMETER project ( https://www.openmeter.de/ ) which grants access for analytics as data discovery.
		"
    
    """
    url = f"https://corrently.p.rapidapi.com/alternative/openmeter/meters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "corrently.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

