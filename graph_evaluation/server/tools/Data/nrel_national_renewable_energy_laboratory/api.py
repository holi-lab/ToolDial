import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def nearest_stations(format: str, location: str='433 Bryant St., San Francisco', latitude: str=None, longitude: str=None, radius: str=None, status: str=None, access: str=None, fuel_type: str='ELEC', cards_accepted: str=None, owner_type: str=None, federal_agency_id: str=None, ev_network: str=None, ev_charging_level: str=None, state: str=None, zip: str=None, limit: str='1', offset: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the nearest alternative fuel stations within a distance of a given location."
    location: Type: string Default: None A free-form input describing the address of the location. This may include the address given in a variety of formats, such as:  street, city, state, postal code street, city, state street, postal code postal code city, state (Either the location parameter or both the latitude and longitude parameters are required)
        latitude: Type: decimal Default: -90 to 90 The latitude of the desired location.  (Either the location parameter or both the latitude and longitude parameters are required)
        longitude: Type: decimal Default: -180 to 180 The longitude of the desired location.  (Either the location parameter or both the latitude and longitude parameters are required)
        radius: Type: decimal Default: 5.0 The radius (in miles) around the search location to search for stations within.
        status: Type: string Default: all Options: all, E, P Return stations that match the given status. A single status, or a comma-separated list of multiple statuses, may be given.  Option	Description all	Include both open and planned stations. E	Open: The station is open and carries alternative fuel. P	Planned: The station is not yet open, but plans to carry alternative fuel in the future, or the station is temporarily out of service.
        access: Type: string Default: all Options: all, public, private
        fuel_type: Type: string Default: all Options: all, BD, CNG, E85, ELEC, HY, LNG, LPG Return stations that supply any of the given fuel types. A single fuel type, or a comma-separated list of multiple fuel types, may be given.  Option	Description all	Include all fuel types BD	Biodiesel (B20 and above) CNG	Compressed Natural Gas E85	Ethanol (E85) ELEC	Electric HY	Hydrogen LNG	Liquefied Natural Gas LPG	Liquefied Petroleum Gas (Propane)
        cards_accepted: Type: string Default: all Options: all, A, D, M, V, Cash, Checks, CFN, CleanEnergy, FuelMan, GasCard, PHH, Voyager, Wright_Exp Return stations that accept any of the given payment methods. A single payment method, or a comma-separated list of multiple payment methods, may be given.  Option	Description all	Include all payment types A	American Express D	Discover M	MasterCard V	Visa Cash	 Checks	 CFN	 CleanEnergy	 FuelMan	 GasCard	 PHH	PHH Services Wright_Exp	Wright Express
        owner_type: Type: string Default: all Options: all, FG, LG, P, SG, T Return stations owned by the given types of owners. A single owner type, or a comma-separated list of multiple owner types, may be given.  Option	Description all	Include all owner types P	Privately owned T	Utility owned FG	Federal government owned LG	Local government owned SG	State government owned J	Jointly owned (combination of owner types)
        federal_agency_id: Type: integer Default: all Options: all, (see table to right for IDs) Return stations owned by the given federal agency. A federal agency ID, or a comma-separated list of multiple federal agency IDs, may be given.  Option	Description all	Include all stations, regardless of federal agency owner 3	Defense Agencies 4	U.S. Department of Agriculture 5	Department of Air Force 6	Department of Army 7	Department of Commerce 8	U.S. Department of Energy 9	Department of Health and Human Services 10	Department of Homeland Security 12	Department of Justice 14	Department of Navy 16	Department of the Interior 17	U.S. Department of Transportation 19	Department of Veterans Affairs 20	U.S. Environmental Protection Agency 22	National Aeronautics and Space Administration 23	Smithsonian Institution 24	Social Security Administration 25	U.S. Postal Service 26	United States Marine Corps
        ev_network: Type: string Default: all Options: all, Blink Network, ChargeNet, ChargePoint Network, eVgo Network, EVSE LLC WebNet, RechargeAccess, Rideshare Online, Shorepower Return only electric charging stations that belong to the given network. A single network, or a comma separate list of multiple networks, may be given.  Option all Blink Network ChargeNet ChargePoint Network eVgo Network EVSE LLC WebNet RechargeAccess Rideshare Online Shorepower
        ev_charging_level: Type: string Default: all Options: all, 1, 2, dc_fast Return only electric charging stations that provide the given level of electric vehicle supply equipment (EVSE).  Option	Description all	Include all charging levels 1	Level 1 EVSE (standard 110V outlet) 2	Level 2 EVSE (J1772 connector) dc_fast	DC Fast Charging
        state: Type: string Default: None Return only stations within the given state. State must be given as a two character state code (eg, "CO" for Colorado). A single state, or a comma-separate list of multiple states, may be given.
        zip: Type: string Default: None Return only stations within the given ZIP code. ZIP codes must be exactly 5 digits long. A single ZIP code, or a comma-separate list of multiple ZIP codes, may be given.
        limit: Type: integer Default: None Minimum: 1 The maximum number of results to return.
        offset: Type: integer Default: 0 Minimum: 0 Offset the results found (can be used with the limit parameter to paginate the results).
        
    """
    url = f"https://community-nrel-national-renewable-energy-laboratory.p.rapidapi.com/api/alt-fuel-stations/v1/nearest.{format}"
    querystring = {}
    if location:
        querystring['location'] = location
    if latitude:
        querystring['latitude'] = latitude
    if longitude:
        querystring['longitude'] = longitude
    if radius:
        querystring['radius'] = radius
    if status:
        querystring['status'] = status
    if access:
        querystring['access'] = access
    if fuel_type:
        querystring['fuel_type'] = fuel_type
    if cards_accepted:
        querystring['cards_accepted'] = cards_accepted
    if owner_type:
        querystring['owner_type'] = owner_type
    if federal_agency_id:
        querystring['federal_agency_id'] = federal_agency_id
    if ev_network:
        querystring['ev_network'] = ev_network
    if ev_charging_level:
        querystring['ev_charging_level'] = ev_charging_level
    if state:
        querystring['state'] = state
    if zip:
        querystring['zip'] = zip
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-nrel-national-renewable-energy-laboratory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def utility_rates(format: str, address: str=None, lat: str='37', lon: str='-92', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This service returns annual average utility rates ($/kWH) for residential, commercial and industrial sectors as well as the local utility name for a specific location. This service does not return complex rate information.  Version 3 is the current version of the utility rates API. Previous versions have been deprecated and its users are encouraged to migrate to this newly enhanced version."
    address: Type: string Default: None The address to use (lat/lon returned by Google's geocoding service). Required if lat/lon not provided.
        lat: Type: decimal Default: None Range: -90 to 90 The latitude for the location to use. Required if address not given.
        lon: Type: decimal Default: None Range: -180 to 180 The longitude for the location to use. Required if address not given.
        
    """
    url = f"https://community-nrel-national-renewable-energy-laboratory.p.rapidapi.com/api/utility_rates/v3.{format}"
    querystring = {}
    if address:
        querystring['address'] = address
    if lat:
        querystring['lat'] = lat
    if lon:
        querystring['lon'] = lon
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-nrel-national-renewable-energy-laboratory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_stations(format: str, status: str=None, access: str=None, fuel_type: str='E85,ELEC', cards_accepted: str=None, owner_type: str=None, federal_agency_id: str=None, ev_network: str=None, ev_charging_level: str=None, state: str='CA', zip: str=None, limit: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a full list of alternative fuel stations that match your query."
    status: Type: string Default: all Options: all, E, P Return stations that match the given status. A single status, or a comma-separated list of multiple statuses, may be given.  Option	Description all	Include both open and planned stations. E	Open: The station is open and carries alternative fuel. P	Planned: The station is not yet open, but plans to carry alternative fuel in the future, or the station is temporarily out of service.
        access: Type: string Default: all Options: all, public, private Return stations with the given access type.
        fuel_type: Type: string Default: all Options: all, BD, CNG, E85, ELEC, HY, LNG, LPG Return stations that supply any of the given fuel types. A single fuel type, or a comma-separated list of multiple fuel types, may be given.  Option	Description all	Include all fuel types BD	Biodiesel (B20 and above) CNG	Compressed Natural Gas E85	Ethanol (E85) ELEC	Electric HY	Hydrogen LNG	Liquefied Natural Gas LPG	Liquefied Petroleum Gas (Propane)
        cards_accepted: Type: string Default: all Options: all, A, D, M, V, Cash, Checks, CFN, CleanEnergy, FuelMan, GasCard, PHH, Voyager, Wright_Exp Return stations that accept any of the given payment methods. A single payment method, or a comma-separated list of multiple payment methods, may be given.  Option	Description all	Include all payment types A	American Express D	Discover M	MasterCard V	Visa Cash	 Checks	 CFN	 CleanEnergy	 FuelMan	 GasCard	 PHH	PHH Services Wright_Exp	Wright Express
        owner_type: Type: string Default: all Options: all, FG, LG, P, SG, T Return stations owned by the given types of owners. A single owner type, or a comma-separated list of multiple owner types, may be given.  Option	Description all	Include all owner types P	Privately owned T	Utility owned FG	Federal government owned LG	Local government owned SG	State government owned J	Jointly owned (combination of owner types)
        federal_agency_id: Type: integer Default: all Options: all, (see table to right for IDs) Return stations owned by the given federal agency. A federal agency ID, or a comma-separated list of multiple federal agency IDs, may be given.  Option	Description all	Include all stations, regardless of federal agency owner 3	Defense Agencies 4	U.S. Department of Agriculture 5	Department of Air Force 6	Department of Army 7	Department of Commerce 8	U.S. Department of Energy 9	Department of Health and Human Services 10	Department of Homeland Security 12	Department of Justice 14	Department of Navy 16	Department of the Interior 17	U.S. Department of Transportation 19	Department of Veterans Affairs 20	U.S. Environmental Protection Agency 22	National Aeronautics and Space Administration 23	Smithsonian Institution 24	Social Security Administration 25	U.S. Postal Service 26	United States Marine Corps
        ev_network: Type: string Default: all Options: all, Blink Network, ChargeNet, ChargePoint Network, eVgo Network, EVSE LLC WebNet, RechargeAccess, Rideshare Online, Shorepower Return only electric charging stations that belong to the given network. A single network, or a comma separate list of multiple networks, may be given.  Option all Blink Network ChargeNet ChargePoint Network eVgo Network EVSE LLC WebNet RechargeAccess Rideshare Online Shorepower
        ev_charging_level: Type: string Default: all Options: all, 1, 2, dc_fast Return only electric charging stations that provide the given level of electric vehicle supply equipment (EVSE).  Option	Description all	Include all charging levels 1	Level 1 EVSE (standard 110V outlet) 2	Level 2 EVSE (J1772 connector) dc_fast	DC Fast Charging
        state: Type: string Default: None Return only stations within the given state. State must be given as a two character state code (eg, "CO" for Colorado). A single state, or a comma-separate list of multiple states, may be given.
        zip: Type: string Default: None Return only stations within the given ZIP code. ZIP codes must be exactly 5 digits long. A single ZIP code, or a comma-separate list of multiple ZIP codes, may be given.
        limit: Type: integer Default: None Minimum: 1 The maximum number of results to return. Note: Since results are returned in no specific order, this has limited use, other than for testing purposes.
        
    """
    url = f"https://community-nrel-national-renewable-energy-laboratory.p.rapidapi.com/api/alt-fuel-stations/v1.{format}"
    querystring = {}
    if status:
        querystring['status'] = status
    if access:
        querystring['access'] = access
    if fuel_type:
        querystring['fuel_type'] = fuel_type
    if cards_accepted:
        querystring['cards_accepted'] = cards_accepted
    if owner_type:
        querystring['owner_type'] = owner_type
    if federal_agency_id:
        querystring['federal_agency_id'] = federal_agency_id
    if ev_network:
        querystring['ev_network'] = ev_network
    if ev_charging_level:
        querystring['ev_charging_level'] = ev_charging_level
    if state:
        querystring['state'] = state
    if zip:
        querystring['zip'] = zip
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-nrel-national-renewable-energy-laboratory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pvwatts(system_size: str, lon: str='-105', file_id: str=None, lat: str='40', address: str=None, dataset: str='tmy2', timeframe: str=None, azimuth: str=None, derate: str='0.77', tilt: str=None, tilt_eq_lat: str=None, track_mode: str=None, inoct: str=None, gamma: str=None, callback: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "NREL's PVWatts calculates the energy production of grid-connected photovoltaic (PV) energy systems. This service estimates the performance of hypothetical residential and small commercial PV installations.  Version 4 is the current version of the PVWatts API. Previous versions have been deprecated and its users are encouraged to migrate to this newly enhanced version."
    system_size: Type: decimal Default: None Range: 0.05 to 500000 Nameplate capacity (kW).
        lon: Type: decimal Default: None Range: -180 to 180 The longitude for the location to use. Required if address or file_id not specified.
        file_id: Type: string Default: none An identifier provided by the solar data query web service to specify the climate data file to use. Required if lat/lon or address not specified.
        lat: Type: decimal Default: None Range: -90 to 90 The latitude for the location to use. Required if address or file_id not specified.
        address: Type: string Default: None The address to use (lat/lon returned by Google's geocoding service). Required if lat/lon or file_id not specified.
        dataset: Type: string Default: perez Options: perez, tmy2, tmy3, intl The climate dataset to use. Should not be passed in if using file_id to specify the climate data file.  Option	Description perez	Perez Satellite Solar Resource Data Set tmy2	 TMY2 station data (see Typical Meteorological Year, version 2) tmy3	 TMY3 station data (see Typical Meteorological Year version 3)  intl	 International station data
        timeframe: Type: string Default: monthly Options: monthly, hourly Granularity of the output response.
        azimuth: Type: decimal Default: None Range: 0 to 360 Azimuth angle (degrees).
        derate: Type: decimal Default: None Range: 0 to 1 System derate value.
        tilt: Type: decimal Default: None Tilt angle (degrees).
        tilt_eq_lat: Type: integer Default: 0 Options: 0, 1 Override the tilt variable to equal latitude (default 0 unless tilt provided).  Option	Description 0	False 1	True
        track_mode: Type: integer Default: 1 Options: 0, 1, 2 Tracking mode.  Option	Description 0	Fixed 1	1-Axis 2	2-Axis
        inoct: Type: decimal Default: None Range: 30 to 80 Nominal operating cell temperature (C)
        gamma: Type: decimal Default: None Range: -2 to -0.01 Max power temperature coefficient (%/C)
        callback: Type: string Default: None Return the data using JSONP and the given callback function (only applicable when using the json format).
        
    """
    url = f"https://community-nrel-national-renewable-energy-laboratory.p.rapidapi.com/api/pvwatts/v4.{format}"
    querystring = {'system_size': system_size, }
    if lon:
        querystring['lon'] = lon
    if file_id:
        querystring['file_id'] = file_id
    if lat:
        querystring['lat'] = lat
    if address:
        querystring['address'] = address
    if dataset:
        querystring['dataset'] = dataset
    if timeframe:
        querystring['timeframe'] = timeframe
    if azimuth:
        querystring['azimuth'] = azimuth
    if derate:
        querystring['derate'] = derate
    if tilt:
        querystring['tilt'] = tilt
    if tilt_eq_lat:
        querystring['tilt_eq_lat'] = tilt_eq_lat
    if track_mode:
        querystring['track_mode'] = track_mode
    if inoct:
        querystring['inoct'] = inoct
    if gamma:
        querystring['gamma'] = gamma
    if callback:
        querystring['callback'] = callback
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-nrel-national-renewable-energy-laboratory.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

