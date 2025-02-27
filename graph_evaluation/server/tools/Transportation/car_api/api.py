import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def engines(year: str=None, direction: str='asc', valves: str=None, valve_timing: str=None, fuel_type: str=None, json: str=None, model: str=None, make: str=None, make_model_id: str=None, trim: str=None, cam_type: str=None, engine_type: str=None, make_model_trim_id: str=None, limit: int=None, drive_type: str=None, verbose: str='yes', make_id: str=None, cylinders: str=None, page: int=None, sort: str='id', size: str=None, horsepower_hp: str=None, transmission: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To include additional information about the returned body (such as year, make, model and trim) request with the query parameter as verbose=yes.
		
		For complex queries you may use the json field to send an array of URL encoded JSON conditions, example:
		
		`[{"field": "horsepower_hp", "op": ">=", "val": 100}, {"field": "horsepower_hp", "op": "<=", "val": 300}]`
		
		See /api/vehicle-attributes for a complete list of vehicle attributes.
		
		Allowed operators are: `>`, `<`, `>=`, `<=`, `in`, `not in`, `like`, `not like`, `is null` and `not null`.
		
		Allowed json search fields are: year, make, model, trim, fuel_type, engine_type, transmission, drive_type, cam_type, valve_timing, valves, horsepower_hp, size, cylinders, make_id, make_model_id, and make_model_trim_id."
    verbose: Includes make, model and trim
        
    """
    url = f"https://car-api2.p.rapidapi.com/api/engines"
    querystring = {}
    if year:
        querystring['year'] = year
    if direction:
        querystring['direction'] = direction
    if valves:
        querystring['valves'] = valves
    if valve_timing:
        querystring['valve_timing'] = valve_timing
    if fuel_type:
        querystring['fuel_type'] = fuel_type
    if json:
        querystring['json'] = json
    if model:
        querystring['model'] = model
    if make:
        querystring['make'] = make
    if make_model_id:
        querystring['make_model_id'] = make_model_id
    if trim:
        querystring['trim'] = trim
    if cam_type:
        querystring['cam_type'] = cam_type
    if engine_type:
        querystring['engine_type'] = engine_type
    if make_model_trim_id:
        querystring['make_model_trim_id'] = make_model_trim_id
    if limit:
        querystring['limit'] = limit
    if drive_type:
        querystring['drive_type'] = drive_type
    if verbose:
        querystring['verbose'] = verbose
    if make_id:
        querystring['make_id'] = make_id
    if cylinders:
        querystring['cylinders'] = cylinders
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if size:
        querystring['size'] = size
    if horsepower_hp:
        querystring['horsepower_hp'] = horsepower_hp
    if transmission:
        querystring['transmission'] = transmission
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bodies(make_model_trim_id: str=None, direction: str='asc', year: str=None, page: int=None, verbose: str='yes', json: str=None, make_id: str=None, trim: str=None, sort: str='id', make_model_id: str=None, model: str=None, make: str=None, type: str=None, limit: int=None, doors: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To include additional information about the returned body (such as year, make, model and trim) request with the query parameter as verbose=yes.
		
		For complex queries you may use the json field to send an array of URL encoded JSON conditions, example:
		
		`[{"field": "doors", "op": ">=", "val": 4}, {"field": "type", "op": "in", "val": ["SUV","Van"]}]`
		
		See /api/vehicle-attributes for a complete list of vehicle attributes.
		
		Allowed operators are: `>`, `<`, `>=`, `<=`, `in`, `not in`, `like`, `not like`, `is null` and `not null`.
		
		Allowed json search fields are: year, make, model, trim, type, doors, make_id, make_model_id, and make_model_trim_id."
    verbose: Includes make, model and trim
        
    """
    url = f"https://car-api2.p.rapidapi.com/api/bodies"
    querystring = {}
    if make_model_trim_id:
        querystring['make_model_trim_id'] = make_model_trim_id
    if direction:
        querystring['direction'] = direction
    if year:
        querystring['year'] = year
    if page:
        querystring['page'] = page
    if verbose:
        querystring['verbose'] = verbose
    if json:
        querystring['json'] = json
    if make_id:
        querystring['make_id'] = make_id
    if trim:
        querystring['trim'] = trim
    if sort:
        querystring['sort'] = sort
    if make_model_id:
        querystring['make_model_id'] = make_model_id
    if model:
        querystring['model'] = model
    if make:
        querystring['make'] = make
    if type:
        querystring['type'] = type
    if limit:
        querystring['limit'] = limit
    if doors:
        querystring['doors'] = doors
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def years(make_id: str=None, make: str=None, year: str=None, json: str=None, make_model_id: str=None, model: str=None, trim: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For complex queries you may use the json field to send an array of URL encoded JSON conditions, example:
		
		`[{"field": "make", "op": "in", "val": ["Scion", "Tesla"]}]`
		
		Allowed operators are: `>`, `<`, `>=`, `<=`, `in`, `not in`, `like`, `not like`, `is null` and `not null`.
		
		Allowed search fields are: `year`, `make`, `model`, `trim`, `make_id`, and `make_model_id`."
    
    """
    url = f"https://car-api2.p.rapidapi.com/api/years"
    querystring = {}
    if make_id:
        querystring['make_id'] = make_id
    if make:
        querystring['make'] = make
    if year:
        querystring['year'] = year
    if json:
        querystring['json'] = json
    if make_model_id:
        querystring['make_model_id'] = make_model_id
    if model:
        querystring['model'] = model
    if trim:
        querystring['trim'] = trim
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trims(make_id: str=None, limit: int=None, direction: str='asc', sort: str='id', year: str=None, model: str=None, page: int=None, trim: str=None, make_model_id: str=None, verbose: str='yes', make: str=None, json: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To include additional information about the returned body (such as year, make, model and trim) request with the query parameter as verbose=yes.
		
		For complex queries you may use the json field to send an array of URL encoded JSON conditions, example:
		
		`[{"field": "year", "op": ">=", "val": 2010}, {"field": "year", "op": "<=", "val": 2020}]`
		
		Allowed operators are: `>`, `<`, `>=`, `<=`, `in`, `not in`, `like`, `not like`, `is null` and `not null`.
		
		Allowed json search fields are: year, make, model, trim, bodies.type, engines.cam_type, engines.cylinders, engines.drive_type, engines.engine_type, engines.fuel_type, engines.transmission, engines.valve_timing, engines.valves, make_id, make_model_id, make_model_trim_id, created, and modified."
    verbose: Includes make, model and trim
        
    """
    url = f"https://car-api2.p.rapidapi.com/api/trims"
    querystring = {}
    if make_id:
        querystring['make_id'] = make_id
    if limit:
        querystring['limit'] = limit
    if direction:
        querystring['direction'] = direction
    if sort:
        querystring['sort'] = sort
    if year:
        querystring['year'] = year
    if model:
        querystring['model'] = model
    if page:
        querystring['page'] = page
    if trim:
        querystring['trim'] = trim
    if make_model_id:
        querystring['make_model_id'] = make_model_id
    if verbose:
        querystring['verbose'] = verbose
    if make:
        querystring['make'] = make
    if json:
        querystring['json'] = json
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def interior_colors(model: str=None, name: str=None, trim: str=None, page: int=None, direction: str='asc', limit: int=None, make_model_trim_id: str=None, year: str=None, rgb: str=None, sort: str='id', verbose: str='yes', json: str=None, make_id: str=None, make: str=None, make_model_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To include additional information about the returned body (such as year, make, model and trim) request with the query parameter as verbose=yes.
		
		For complex queries you may use the json field to send an array of URL encoded JSON conditions, example:
		
		[{"field": "name", "op": "in", "val": ["red", "blue"]}]
		
		Allowed operators are: `>`, `<`, `>=`, `<=`, `in`, `not in`, `like`, `not like`, `is null` and `not null`.
		
		Allowed json search fields are: year, make, model, trim, name, rgb, make_id, make_model_id, and make_model_trim_i"
    verbose: Includes make, model and trim
        
    """
    url = f"https://car-api2.p.rapidapi.com/api/interior-colors"
    querystring = {}
    if model:
        querystring['model'] = model
    if name:
        querystring['name'] = name
    if trim:
        querystring['trim'] = trim
    if page:
        querystring['page'] = page
    if direction:
        querystring['direction'] = direction
    if limit:
        querystring['limit'] = limit
    if make_model_trim_id:
        querystring['make_model_trim_id'] = make_model_trim_id
    if year:
        querystring['year'] = year
    if rgb:
        querystring['rgb'] = rgb
    if sort:
        querystring['sort'] = sort
    if verbose:
        querystring['verbose'] = verbose
    if json:
        querystring['json'] = json
    if make_id:
        querystring['make_id'] = make_id
    if make:
        querystring['make'] = make
    if make_model_id:
        querystring['make_model_id'] = make_model_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exterior_colors(trim: str=None, make_model_id: str=None, sort: str='id', verbose: str='yes', rgb: str=None, limit: int=None, direction: str='asc', name: str=None, make_id: str=None, make: str=None, year: str=None, model: str=None, make_model_trim_id: str=None, page: int=None, json: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To include additional information about the returned body (such as year, make, model and trim) request with the query parameter as verbose=yes.
		
		For complex queries you may use the json field to send an array of URL encoded JSON conditions, example:
		
		[{"field": "name", "op": "in", "val": ["red", "blue"]}]
		
		Allowed operators are: `>`, `<`, `>=`, `<=`, `in`, `not in`, `like`, `not like`, `is null` and `not null`.
		
		Allowed json search fields are: year, make, model, trim, name, rgb, make_id, make_model_id, and make_model_trim_i"
    verbose: Includes make, model and trim
        
    """
    url = f"https://car-api2.p.rapidapi.com/api/exterior-colors"
    querystring = {}
    if trim:
        querystring['trim'] = trim
    if make_model_id:
        querystring['make_model_id'] = make_model_id
    if sort:
        querystring['sort'] = sort
    if verbose:
        querystring['verbose'] = verbose
    if rgb:
        querystring['rgb'] = rgb
    if limit:
        querystring['limit'] = limit
    if direction:
        querystring['direction'] = direction
    if name:
        querystring['name'] = name
    if make_id:
        querystring['make_id'] = make_id
    if make:
        querystring['make'] = make
    if year:
        querystring['year'] = year
    if model:
        querystring['model'] = model
    if make_model_trim_id:
        querystring['make_model_trim_id'] = make_model_trim_id
    if page:
        querystring['page'] = page
    if json:
        querystring['json'] = json
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def makes(limit: int=None, direction: str='asc', sort: str='id', page: int=None, make: str=None, year: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search makes by name and year."
    
    """
    url = f"https://car-api2.p.rapidapi.com/api/makes"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if direction:
        querystring['direction'] = direction
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if make:
        querystring['make'] = make
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vin_decoder(vin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Decodes Vehicle Identification Numbers. The result will include a list of specifications in the specs property and a list of all possible trims matching the VIN in the trims property."
    
    """
    url = f"https://car-api2.p.rapidapi.com/api/vin/{vin}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def models(limit: int=None, make_id: str=None, year: str=None, page: int=None, sort: str='id', make: str=None, model: str=None, direction: str='asc', verbose: str='yes', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search models by year, make, model, trim or make_id.
		
		To include the models make in the description request with the query parameter as `verbose=yes`.
		
		For complex queries you may use the json field to send an array of URL encoded JSON conditions, example:
		
		`[{"field": "make", "op": "in", "val": ["Ford", "Acura"]}, {"field": "year", "op": ">=", "val": 2010}]
		
		Allowed json operators are: =, !=, >, <, >=, <=, in, not in, like, not like, not null, and is null.
		
		Allowed json search fields are: year, make, model, make_id, created, and modified."
    verbose: Includes make, model and trim
        
    """
    url = f"https://car-api2.p.rapidapi.com/api/models"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if make_id:
        querystring['make_id'] = make_id
    if year:
        querystring['year'] = year
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if make:
        querystring['make'] = make
    if model:
        querystring['model'] = model
    if direction:
        querystring['direction'] = direction
    if verbose:
        querystring['verbose'] = verbose
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vehicle_attributes(attribute: str='bodies.type', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all options for given attribute."
    attribute: The attribute options to be returned
        
    """
    url = f"https://car-api2.p.rapidapi.com/api/vehicle-attributes"
    querystring = {}
    if attribute:
        querystring['attribute'] = attribute
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mileages(make_model_id: str=None, limit: int=None, make_model_trim_id: str=None, trim: str=None, json: str=None, range_highway: str=None, sort: str='id', direction: str='asc', range_city: str=None, page: int=None, combined_mpg: str=None, verbose: str='yes', epa_highway_mpg: str=None, epa_city_mpg: str=None, model: str=None, year: str=None, make_id: str=None, make: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To include additional information about the returned body (such as year, make, model and trim) request with the query parameter as verbose=yes.
		
		For complex queries you may use the json field to send an array of URL encoded JSON conditions, example:
		
		[{"field": "combined_mpg", "op": ">=", "val": 20}, {"field": "combined_mpg", "op": "<=", "val": 30}]
		
		Allowed operators are: `>`, `<`, `>=`, `<=`, `in`, `not in`, `like`, `not like`, `is null` and `not null`.
		
		Allowed json search fields are: year, make, model, trim, combined_mpg, epa_city_mpg, epa_highway_mpg, range_city, range_highway, make_id, make_model_id, and make_model_trim_id."
    verbose: Includes make, model and trim
        
    """
    url = f"https://car-api2.p.rapidapi.com/api/mileages"
    querystring = {}
    if make_model_id:
        querystring['make_model_id'] = make_model_id
    if limit:
        querystring['limit'] = limit
    if make_model_trim_id:
        querystring['make_model_trim_id'] = make_model_trim_id
    if trim:
        querystring['trim'] = trim
    if json:
        querystring['json'] = json
    if range_highway:
        querystring['range_highway'] = range_highway
    if sort:
        querystring['sort'] = sort
    if direction:
        querystring['direction'] = direction
    if range_city:
        querystring['range_city'] = range_city
    if page:
        querystring['page'] = page
    if combined_mpg:
        querystring['combined_mpg'] = combined_mpg
    if verbose:
        querystring['verbose'] = verbose
    if epa_highway_mpg:
        querystring['epa_highway_mpg'] = epa_highway_mpg
    if epa_city_mpg:
        querystring['epa_city_mpg'] = epa_city_mpg
    if model:
        querystring['model'] = model
    if year:
        querystring['year'] = year
    if make_id:
        querystring['make_id'] = make_id
    if make:
        querystring['make'] = make
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trim_view(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all data associated with the vehicle trim."
    
    """
    url = f"https://car-api2.p.rapidapi.com/api/trims/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

