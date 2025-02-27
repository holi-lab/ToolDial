import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_census_tract_by_location(latitude: int=37, longitude: int=-77, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve Census Tract by Location"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/censustract/within"
    querystring = {}
    if latitude:
        querystring['latitude'] = latitude
    if longitude:
        querystring['longitude'] = longitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_census_tract_by_geoid(geoid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Census Tract by GeoId"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/censustract/{geoid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_census_tract_by_county(county: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a GeoJson resprentation of CensusTracts by county"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/censustract/county/{county}/state/{state}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_county_names_by_state(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of county names by state acronym"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/censustract/state/{state}/county/names"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_state_boundaries(nameabbrv: str, accept: str='application/topojson', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for State Boundary. this will return a simpler GeoJson representation of the State, and will not include zip codes only counties that make up that state as feature properties"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/state/{nameabbrv}"
    querystring = {}
    if accept:
        querystring['Accept'] = accept
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def depreciated_query_by_zipcodes_boundaries(zipcode: str='22066,20003,20019,20015,20854', city: str=None, showcenter: bool=None, state: str=None, showdetails: bool=None, is_and: bool=None, combine: bool=None, county: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query by Zipcodes Boundaries"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary"
    querystring = {}
    if zipcode:
        querystring['zipcode'] = zipcode
    if city:
        querystring['city'] = city
    if showcenter:
        querystring['showCenter'] = showcenter
    if state:
        querystring['state'] = state
    if showdetails:
        querystring['showDetails'] = showdetails
    if is_and:
        querystring['and'] = is_and
    if combine:
        querystring['combine'] = combine
    if county:
        querystring['county'] = county
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mega_plan_only_query_for_dma_region_area_boundary(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "*MEGA Subscription required*
		Query for Nielsen's DMA Region Boundary and metadata by DMA Name.
		The metadata  data is publicly available from thevab.com  from URL  https://thevab.com/storage/app/media/Toolkit/DMA_Map_2019.pdf 
		the boundaries were created internally in boundaries-io.com and the US census files.
		
		The result includes all counties and zip codes in the DMA  region, with DMA Ranking from 2020 located here:
		https://mediatracks.com/resources/nielsen-dma-rankings-2020/
		
		For more granular data contact www.nielsen.com"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/dma"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mega_plan_only_query_for_dma_region_names_by_state(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "*MEGA Subscription required*
		Query for DMA Region Names by State"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/dma/state/{state}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_a_a_valid_h3_address_as_a_long_datatype_value_into_a_h3_address_hex(h3index: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert a a valid H3 Address as a Long Datatype value into a  H3 Address Hex"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/rest/v1/public/h3/convert/{h3index}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_state_or_territories(statename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for any of the 57 state and territories by abbreviation.
		
		List of  abbreviation:
		[https://secure.ssa.gov/poms.nsf/lnx/0901501010](url)"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/territories/{statename}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_united_states_outline(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Outline of US and it's territories."
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/unitedstates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_h3_hexagon_by_h3_index_hex(h3ndex: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for Hexagon by H3 Index(hex)"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/h3/zipcode/index/{h3ndex}/hex"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_h3_hexagon_by_location(latitude: int, longitude: int, resolution: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for H3 Hexagon by Location"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/h3/zipcode/location/boundary"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    if resolution:
        querystring['resolution'] = resolution
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_by_location_for_place_boundaries(longitude: int, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query by Location for Place/City boundaries.
		Incorporated and unincorporated Places.
		
		This is from the US Census Places shapefiles."
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/place/within"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_place_names_by_state(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for Place Names by State"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/place/state/{state}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_by_location_for_neighborhood_boundaries(longitude: str, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query by Location for Neighborhood"
    longitude: longitude
        latitude: latitude
        
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/neighborhood/within"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_neighborhood_by_state_with_city_county(city: str=None, state: str='va', name: str='Rock Spring', nameonly: bool=None, county: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for Neighborhood  by State with City, County"
    city: City
        state: State
        name: Neighborhood Name
        nameonly: return a JSON array of neighborhood names(a Query by only the State is allowed)
        county: County
        
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/neighborhood"
    querystring = {}
    if city:
        querystring['city'] = city
    if state:
        querystring['state'] = state
    if name:
        querystring['name'] = name
    if nameonly:
        querystring['nameOnly'] = nameonly
    if county:
        querystring['county'] = county
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_neighborhood_by_name_and_state(name: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for Neighborhood by Name and State"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/neighborhood/{name}/state/{state}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_by_zipcode_s_for_zipcode_bondaries(zipcode: str='22066,20003,20019,20015,20854', city: str=None, county: str=None, combine: bool=None, showdetails: bool=None, is_and: bool=None, showcenter: bool=None, state: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get US State,Zipcode, or City boundaries"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/zipcode"
    querystring = {}
    if zipcode:
        querystring['zipcode'] = zipcode
    if city:
        querystring['city'] = city
    if county:
        querystring['county'] = county
    if combine:
        querystring['combine'] = combine
    if showdetails:
        querystring['showDetails'] = showdetails
    if is_and:
        querystring['and'] = is_and
    if showcenter:
        querystring['showCenter'] = showcenter
    if state:
        querystring['state'] = state
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_by_a_fips_for_zipcode_boundaries(fips: int, showdetails: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "FIPS county code. The FIPS county code is a five-digit Federal Information Processing Standards (FIPS) code (FIPS 6-4) which uniquely identifies counties and county equivalents in the United States."
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/zipcode/fips/{fips}"
    querystring = {}
    if showdetails:
        querystring['showDetails'] = showdetails
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_by_radius_in_miles_for_zipcode_boundaries(longitude: int, latitude: int, showcirlce: bool=None, radius: int=1, showdetails: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get US zipcode boundaries that are contained within OR intersect the radius in miles of the point(latitude,longtitude)."
    longitude: radius size in miles. defaults to 1 mile.
        latitude: latitude of radius center.
        showcirlce: show the bounding Circle in results or not.
        radius: radius size in miles. defaults to 1 mile.
        showdetails: Show County & State FIPS
        
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/zipcode/location"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    if showcirlce:
        querystring['showCirlce'] = showcirlce
    if radius:
        querystring['radius'] = radius
    if showdetails:
        querystring['showDetails'] = showdetails
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_by_location_for_zipcode_boundaries(longitude: int, latitude: int, showwithinpoint: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for a zipcode that intersect a location."
    longitude: longitude
        latitude: latitude
        showwithinpoint: Show the location(lat,long) used as a GeoJsonPoint in results.
        
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/zipcode/within"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    if showwithinpoint:
        querystring['showwithinpoint'] = showwithinpoint
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_county_names_by_state(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for County Names By State"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/county/{state}/names"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_by_for_county_boundaries(countyname: str, stateabbrv: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the Generic County Boundary file from US Census."
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/county/{countyname}/state/{stateabbrv}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_neighborhood_names_by_state(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for Neighborhood Names by State"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/neighborhood/state/{state}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_by_location_for_county_boundaries(longitude: int, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query by Location for a County boundary"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/county/within"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_place_boundaries(stateabbrv: str, name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for Place/City boundaries
		
		Incorporated and unincorporated Places."
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/place/{name}/state/{stateabbrv}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_by_location_for_state_boundaries(longitude: int, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query by Location for State boundaries"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/reaperfire/rest/v1/public/boundary/state/within"
    querystring = {'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_h3_hexagon_by_h3_index_long(h3ndex: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for H3 Hexagon by H3 Index(Long)"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/h3/zipcode/index/{h3ndex}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query_for_distance_bearing_between_two_h3_index_hex(h3index2: str, h3index1: str, showlinebetween: bool=None, showpoints: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query for Distance & Bearing Between H3 Index(hex)"
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/rest/v1/public/boundary/h3/zipcode/index1/{h3index1}/index2/{h3index2}/hex"
    querystring = {}
    if showlinebetween:
        querystring['showLineBetween'] = showlinebetween
    if showpoints:
        querystring['showPoints'] = showpoints
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_a_a_valid_h3_address_hex_value_into_a_h3_address_as_a_long_datatype(h3index: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert a a valid H3 Address Hex value into a H3 Address as a Long Datatype."
    
    """
    url = f"https://vanitysoft-boundaries-io-v1.p.rapidapi.com/rest/v1/public/h3/convert/{h3index}/hex"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vanitysoft-boundaries-io-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

