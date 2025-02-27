import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def zones_type_zoneid_forecast(type: str, zoneid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Forecast data for zone. Example: /zones/forecast/MOZ028/forecast"
    type: type: a valid zone type (list forthcoming)
        zoneid: zoneId: a zone id (list forthcoming)
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/zones/{type}/{zoneid}/forecast"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_types_typeid_locations_locationid(locationid: str, typeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A product for a location that has issued a product for a specific type.  Example: /products/types/AFD/locations/EAX"
    locationid: locationId: an id of a valid location (list forthcoming)
        typeid: typeId: an id of a valid product type 
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/products/types/{typeid}/locations/{locationid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_types_typeid(typeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of products by type. Example: /products/types/AFD"
    typeid: typeId: an id of a valid product type 
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/products/types/{typeid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def points_point_forecast_hourly(point: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hourly forecast data for a point. This response is derrived from the /gridpoints endpoint and is intentionally less structured. If more structure is required, developers should use the /gridpoints endpoint directly. Example: /points/39.0693,-94.6716/forecast/hourly"
    point: point: EPSG:4326 latitude, EPSG:4326 longitude
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/points/{point}/forecast/hourly"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gridpoints_wfo_x_y(wfo: str, x: str, y: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Raw (commonly referred to as "gridded") data provided by the Weather Office. Every forecast request will use this data to build the forecast response. The grid for a given location is determined by the "property.forecastGridData" property in the /points/{lat},{lon} endpoint. Example: /gridpoints/EAX/40,48"
    wfo: wfo: a weather office id (https://en.wikipedia.org/wiki/List_of_National_Weather_Service_Weather_Forecast_Offices)
        x: x: the grid x coordinate
        y: y: the grid y coordinate
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/gridpoints/{wfo}/{x},{y}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts_active_zone_zoneid(zoneid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of active alerts by zone id. The ATOM format returns items in CAP-ATOM. Example: /alerts/active/zone/ILZ081"
    zoneid: zoneId: a valid zone, see list in counts endpoint
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/alerts/active/zone/{zoneid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_productid(productid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Data of a product.Example: /product/NWS-IDP-PROD-2202326-2064512 (this id is likely now an expired product)"
    productid: productId: an id provided by from another /product endpoint
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/products/{productid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stations_stationid_observations(stationid: str, end: str=None, limit: str=None, start: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of observations for a station.  NOTE! The API uses MADIS to provide observation data. The observation may be delayed while MADIS completes quality checks of the data. To limit the delay, the API is provided with incremental updates with various levels of checked data, and the API will return the variation of the observation with the most checked data. It is possible that no data is provided on the first variation, or that no data is checked on the first variation. It is up to the consumer to determine if the quality of data is appropriate. If not, the previous observation can be requested, or the next nearest station can be used. The API returns the state of the data for each value in the response, where "Z" (or "null") is for not checked (and may never be for some values) and "V" for checked. Please refer to MADIS for complete documentation on their data quality process.  Example: /stations/KMKC/observations"
    stationid: stationId: a valid station id (e.g. as provided by the /points/{point}/stations endpoint)
        end:  End time (ISO8601DateTime)
        limit: Limit (an integer)
        start: Start time (ISO8601DateTime)
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/stations/{stationid}/observations"
    querystring = {}
    if end:
        querystring['end'] = end
    if limit:
        querystring['limit'] = limit
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of product types with an active product"
    
    """
    url = f"https://national-weather-service.p.rapidapi.com/products/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def points_point_stations(point: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Stations nearest to a point in order of distance. Example: /points/39.0693,-94.6716/stations"
    point: point: EPSG:4326 latitude, EPSG:4326 longitude
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/points/{point}/stations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts_alertid(alertid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A specific alert by id provided by a search or list. Example: /alerts/NWS-IDP-PROD-2202530-2064731"
    
    """
    url = f"https://national-weather-service.p.rapidapi.com/alerts/{alertid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts_active_count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of active counts for regions, areas and zones. A list of these items forthcoming."
    
    """
    url = f"https://national-weather-service.p.rapidapi.com/alerts/active/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_locations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of locations with active products."
    
    """
    url = f"https://national-weather-service.p.rapidapi.com/products/locations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts_active(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of active alerts that can be filtered by parameters. Uses same parameters as the /alerts endpoint, but sets "active" parameter to 1 and ignores "start" and "end" parameters. The ATOM format returns items in CAP-ATOM."
    
    """
    url = f"https://national-weather-service.p.rapidapi.com/alerts/active"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zones_type_zoneid(type: str, zoneid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Metadata for a zone. Example /zones/forecast/MOZ028"
    type: type: a valid zone type (list forthcoming)
        zoneid: zoneId: a zone id (list forthcoming)
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/zones/{type}/{zoneid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts_active_area_area(area: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of active alerts by area. The ATOM format returns items in CAP-ATOM."
    area: area: a valid area, see list in counts endpoint
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/alerts/active/area/{area}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def points_point_forecast(point: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Forecast data for a point. The DWML format is a temporary format to aid transition and will be sunset at a later date. This response is derrived from the /gridpoints endpoint and is intentionally less structured. If more structure is required, developers should use the /gridpoints endpoint directly. Example: /points/39.0693,-94.6716/forecast"
    point: point: EPSG:4326 latitude, EPSG:4326 longitude
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/points/{point}/forecast"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_locations_locationid_types(locationid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Metadata about a Weather Office. Example /offices/EAX"
    locationid: officeId: a weather office id (https://en.wikipedia.org/wiki/List_of_National_Weather_Service_Weather_Forecast_Offices)
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/products/locations/{locationid}/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stations_stationid_observations_recordid(stationid: str, recordid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Data for a specific observation.  NOTE! See note in /stations/{stationId}/observations for important details on observation data.  Example: /stations/KMKC/observations/2017-01-04T18:54:00+00:00"
    stationid: stationsId: Station id
        recordid: recordId, Record Id (ISO8601DateTime)
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/stations/{stationid}/observations/{recordid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stations(limit: int=None, states: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of stations and station metadata that can be filtered by parameters. If no parameters are provided, then all stations are returned. This list is not configured by field offices and only returns active stations. Example: /stations?limit=10&states=KS,MO"
    limit: Limit the Results
        states: Filter by States (by abbreviation)
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/stations"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if states:
        querystring['states'] = states
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts_parameters(start: str=None, end: str=None, status: str=None, zone_type: str=None, active: str=None, type: str=None, point: str=None, state: str=None, zone: str=None, urgency: str=None, region: str=None, certainty: str=None, severity: str=None, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of alerts that can be filtered by parameters. If no parameters are provided, then all alerts are returned. The ATOM format returns items in CAP-ATOM."
    start: start, Start time (ISO8601DateTime)
        end: end, End time (ISO8601DateTime)
        status: status, Event status (alert, update, cancel)
        zone_type: zone_type, Zone type (land or marine)
        active: active, Active alerts (1 or 0)
        type: type, Event type (list forthcoming)
        point: point, Point (latitude,longitude)
        state: State/marine code (list forthcoming)
        zone: zone, Zone Id (forecast or county, list forthcoming)
        urgency: urgency, Urgency (expected, immediate)
        region: region, Region code (list forthcoming)
        certainty: certainty, Certainty (likely, observed)
        severity: severity, Severity (minor, moderate, severe)
        limit:  limit, Limit (an integer)
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/alerts"
    querystring = {}
    if start:
        querystring['start'] = start
    if end:
        querystring['end'] = end
    if status:
        querystring['status'] = status
    if zone_type:
        querystring['zone_type'] = zone_type
    if active:
        querystring['active'] = active
    if type:
        querystring['type'] = type
    if point:
        querystring['point'] = point
    if state:
        querystring['state'] = state
    if zone:
        querystring['zone'] = zone
    if urgency:
        querystring['urgency'] = urgency
    if region:
        querystring['region'] = region
    if certainty:
        querystring['certainty'] = certainty
    if severity:
        querystring['severity'] = severity
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts_active_region_region(region: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of active alerts by region. The ATOM format returns items in CAP-ATOM. Example: /alerts/active/region/GL"
    region: area: a valid region, see list in counts endpoint
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/alerts/active/region/{region}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def points_point(point: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Metadata about a point. This is the primary endpoint for forecast information for a location. It contains linked data for the forecast, the hourly forecast, observation and other information. Example: /points/39.0693,-94.6716"
    point: point: EPSG:4326 latitude, EPSG:4326 longitude
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/points/{point}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def products_types_typeid_locations(typeid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of locations that have issues products for a type. Example: /products/types/AFD/locations"
    typeid: typeId: an id of a valid product type (list forthcoming)
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/products/types/{typeid}/locations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stations_stationid_observations_current(stationid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The most current observation for a station. Due to a legacy requirement, this endpoint will support XML for the near future when using the Accept header. It is highly recommend that applications update to the JSON format.  NOTE! See note in /stations/{stationId}/observations for important details on observation data.  Example: /stations/KMKC/observations/current"
    stationid: stationId: Station Id (e.g. as provided by the /points/{point}/stations endpoint)
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/stations/{stationid}/observations/current"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stations_stationid(stationid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Metadata about a station. Similar to /stations endpoint with id parameter. Example: /stations/KMKC"
    stationid: stationId: the id of a station from the /points/{point}/stations endpoint
        
    """
    url = f"https://national-weather-service.p.rapidapi.com/stations/{stationid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "national-weather-service.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

