import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v2_0_solar_estimation(lat: int, wp: int, lon: int, deg: int=34, loss: int=14, az: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "# Solar Power Estimation
		Estimated photovoltaic (PV) system performance per month of an installation with given characteristics."
    wp: Installed Watt-Peak (eq. kWp/1000)
        deg: Plant declination (tilt) degrees, 0 (horizontal) … 90 (vertical)
        az: Plant azimuth, facing direction relative to south:  -180 … 180 (-180 = north, -90 = east, 0 = south, 90 = west, 180 = north)
        
    """
    url = f"https://solarenergyprediction.p.rapidapi.com/v2.0/solar/estimation"
    querystring = {'lat': lat, 'wp': wp, 'lon': lon, }
    if deg:
        querystring['deg'] = deg
    if loss:
        querystring['loss'] = loss
    if az:
        querystring['az'] = az
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "solarenergyprediction.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_0_solar_prediction(tech: str=None, lon: int=8, plant: str='0xAb425D74eA0113d11c3092E47a4C0542C25973db', az: int=45, deg: int=32, decoration: str='default', inverter: int=4500, wp: int=6442, lat: int=49, zip: str='69256', loss: int=14, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "## Solar Power Forecasting 
		The PV generation forecast for unmetered PV systems makes it possible to predict production of a PV system on an hourly basis for the next 4 days without having your own metering equipment or system-specific generation data. 
		
		**Key Features**
		- based on quality weather forecast
		- solar irradiance prediction based on real-time satellite data
		- support of multiple strings, slopes, system losses and azimuths
		- hourly forecast up to 96 hours (4 days)
		- No AI training required. 
		- Ready for usage in Energy Management Systems 
		
		
		**Usage**
		Specify all individual parameters of a pv-plant with query parameters. The result will give a unique plant ID, that might be used in future requests to reference parameters using `plant` query parameter.
		
		[![Run in Postman](https://run.pstmn.io/button.svg)](https://god.gw.postman.com/run-collection/15081091-0b194e23-bda0-408d-b732-03239a67af4e?action=collection%2Ffork&collection-url=entityId%3D15081091-0b194e23-bda0-408d-b732-03239a67af4e%26entityType%3Dcollection%26workspaceId%3D62346650-ac84-4b89-9975-fb61337ec7de#?env%5BRapidAPI%5D=W3sia2V5IjoiWC1SYXBpZEFQSS1LZXkiLCJ2YWx1ZSI6ImQ0MGQyZTA0MWJtc2g5ZGZmN2ZmZThmOTMxOTVwMWRiNjU2anNuYTMyN2Y0NzljMGMxIiwiZW5hYmxlZCI6dHJ1ZSwidHlwZSI6InNlY3JldCIsInNlc3Npb25WYWx1ZSI6ImQ0MGQyZTA0MWJtc2g5ZGZmN2ZmZThmOTMxOTVwMWRiNjU2anNuYTMyN2Y0NzljMGMxIiwic2Vzc2lvbkluZGV4IjowfV0=)"
    tech: Panel technology used (for compatibility with PVGIS).
        lon: Geocode longitude part of pv plant  (required if parameter plant or zip is not given).
        plant: If specified all other query parameters will be taken from the stored  value and must not be given with the query string. 

Use as comma separated list of IDs to retrieve a forecast for multiple specifications.
        az: plane azimuth, -180 … 180 (-180 = north, -90 = east, 0 = south, 90 = west, 180 = north)
        deg: plane declination degrees, 0 (horizontal) … 90 (vertical)
        decoration: Allows to use SolarPredictionAPI as a plug-in replacement for other APIs.

Supported decorations:
| `default`  | Output will be formatted as shown in example response |
| `forecast.solar` | Output will be formatted like [https://forecast.solar](http://doc.forecast.solar/doku.php?id=api:estimate#example) |
|----|----|
        inverter: Maximum inverter capacity in Watt. 

If specified, the prediction will be limited to this amount.
        wp: Installed Watt-Peak  (eq. kWp/1000)
        lat: (required if parameter plant or zip is not given).
        zip: Postalcode (only in Germany) - Might be used instead of lat/lon geolocation.
        
    """
    url = f"https://solarenergyprediction.p.rapidapi.com/v2.0/solar/prediction"
    querystring = {}
    if tech:
        querystring['tech'] = tech
    if lon:
        querystring['lon'] = lon
    if plant:
        querystring['plant'] = plant
    if az:
        querystring['az'] = az
    if deg:
        querystring['deg'] = deg
    if decoration:
        querystring['decoration'] = decoration
    if inverter:
        querystring['inverter'] = inverter
    if wp:
        querystring['wp'] = wp
    if lat:
        querystring['lat'] = lat
    if zip:
        querystring['zip'] = zip
    if loss:
        querystring['loss'] = loss
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "solarenergyprediction.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

