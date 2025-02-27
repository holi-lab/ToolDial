import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def status(departureairport: str=None, aircraftregistrationnumber: str=None, flighttype: str=None, carriercode: str=None, codetype: str=None, version: str='v2', arrivalairport: str=None, departuredatetime: str=None, after: str=None, arrivaldatetime: str=None, servicetype: str=None, flightnumber: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get flight status information by specifying either the Departure Date or Arrival Date parameter value (mandatory), and other parameters such as Flight Number, Carrier Code, Departure, and Arrival Airport (non-mandatory)."
    departureairport: A code assigned by the IATA/ICAO/FAA to an airport location representing departure airport. If CodeType not specified, airports are searched by IATA (default). Accepts comma separated values.
        aircraftregistrationnumber: Aircraft registration number (Tail Number).
        flighttype: Type of flight: Scheduled, Unscheduled, General Aviation (GA). Accepts comma separated values.
        carriercode: Carrier (airline) code assigned either by IATA or ICAO.
        codetype: Code type for Carriers and Airports. If not specified, carriers and airports are searched by IATA (default). Code type values are IATA, ICAO - for carriers and airports, FAA - for airports. Accepts comma separated values.
        version: Api version.
        arrivalairport: A code assigned by the IATA/ICAO/FAA to an airport location representing arrival airport. If CodeType not specified, airports are searched by IATA (default). Accepts comma separated values.
        departuredatetime: Departure date and/or time or date and/or time range is local time in ISO 8601 date format (e.g. YYYY-MM-DD[THH:MM] or YYYY-MM-DD[THH:MM]/YYYY-MM-DD[THH:MM]). The maximum date range is 31 days.
        after: A cursor after refers to a random string of characters which marks a specific item in a list of data.
        arrivaldatetime: Arrival date and/or time or date and/or time range is local time in ISO 8601 date format (e.g. YYYY-MM-DD[THH:MM] or YYYY-MM-DD[THH:MM]/YYYY-MM-DD[THH:MM]). The maximum date range is 31 days.
        servicetype: Type of service: Passenger, Cargo. Only flights for the specified service type are returned if specified. Accepts comma separated values.
        flightnumber: A numeric part (up to four digits) of a flight designator.
        
    """
    url = f"https://flight-info-api.p.rapidapi.com/status"
    querystring = {}
    if departureairport:
        querystring['DepartureAirport'] = departureairport
    if aircraftregistrationnumber:
        querystring['AircraftRegistrationNumber'] = aircraftregistrationnumber
    if flighttype:
        querystring['FlightType'] = flighttype
    if carriercode:
        querystring['CarrierCode'] = carriercode
    if codetype:
        querystring['CodeType'] = codetype
    if version:
        querystring['version'] = version
    if arrivalairport:
        querystring['ArrivalAirport'] = arrivalairport
    if departuredatetime:
        querystring['DepartureDateTime'] = departuredatetime
    if after:
        querystring['After'] = after
    if arrivaldatetime:
        querystring['ArrivalDateTime'] = arrivaldatetime
    if servicetype:
        querystring['ServiceType'] = servicetype
    if flightnumber:
        querystring['FlightNumber'] = flightnumber
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-info-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def schedules(after: str=None, codetype: str=None, carriercode: str=None, arrivaldatetime: str=None, servicetype: str=None, aircraftregistrationnumber: str=None, flighttype: str=None, arrivalairport: str=None, departureairport: str=None, flightnumber: int=None, version: str='v2', departuredatetime: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get flight schedule information by specifying either the Departure Date or Arrival Date parameter value (mandatory), and other parameters such as Flight Number, Carrier Code, Departure, and Arrival Airport (non-mandatory)."
    after: A cursor after refers to a random string of characters which marks a specific item in a list of data.
        codetype: Code type for Carriers and Airports. If not specified, carriers and airports are searched by IATA (default). Code type values are IATA, ICAO - for carriers and airports, FAA - for airports. Accepts comma separated values.
        carriercode: Carrier (airline) code assigned either by IATA or ICAO.
        arrivaldatetime: Arrival date and/or time or date and/or time range is local time in ISO 8601 date format (e.g. YYYY-MM-DD[THH:MM] or YYYY-MM-DD[THH:MM]/YYYY-MM-DD[THH:MM]). The maximum date range is 31 days.
        servicetype: Type of service: Passenger, Cargo. Only flights for the specified service type are returned if specified. Accepts comma separated values.
        aircraftregistrationnumber: Aircraft registration number (Tail Number).
        flighttype: Type of flight: Scheduled, Unscheduled, General Aviation (GA). Accepts comma separated values.
        arrivalairport: A code assigned by the IATA/ICAO/FAA to an airport location representing arrival airport. If CodeType not specified, airports are searched by IATA (default). Accepts comma separated values.
        departureairport: A code assigned by the IATA/ICAO/FAA to an airport location representing departure airport. If CodeType not specified, airports are searched by IATA (default). Accepts comma separated values.
        flightnumber: A numeric part (up to four digits) of a flight designator.
        version: Api version.
        departuredatetime: Departure date and/or time or date and/or time range is local time in ISO 8601 date format (e.g. YYYY-MM-DD[THH:MM] or YYYY-MM-DD[THH:MM]/YYYY-MM-DD[THH:MM]). The maximum date range is 31 days.
        
    """
    url = f"https://flight-info-api.p.rapidapi.com/schedules"
    querystring = {}
    if after:
        querystring['After'] = after
    if codetype:
        querystring['CodeType'] = codetype
    if carriercode:
        querystring['CarrierCode'] = carriercode
    if arrivaldatetime:
        querystring['ArrivalDateTime'] = arrivaldatetime
    if servicetype:
        querystring['ServiceType'] = servicetype
    if aircraftregistrationnumber:
        querystring['AircraftRegistrationNumber'] = aircraftregistrationnumber
    if flighttype:
        querystring['FlightType'] = flighttype
    if arrivalairport:
        querystring['ArrivalAirport'] = arrivalairport
    if departureairport:
        querystring['DepartureAirport'] = departureairport
    if flightnumber:
        querystring['FlightNumber'] = flightnumber
    if version:
        querystring['version'] = version
    if departuredatetime:
        querystring['DepartureDateTime'] = departuredatetime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-info-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

