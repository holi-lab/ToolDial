import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def flightsearch(flnr: str, dtfrom: str=None, dtto: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of flights for a given flight number. Minimum and/or maximum date can optionally be specified to limit the search. Results are ordered by departure date ascending. The next departure time is returned for pagination.
		Will return a 400 if the date range requested is outside the subscription level"
    flnr: The flight number to request
        dtfrom: The minimum date to request
        dtto: The maximum date to request
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/flight/search"
    querystring = {'flnr': flnr, }
    if dtfrom:
        querystring['dtFrom'] = dtfrom
    if dtto:
        querystring['dtTo'] = dtto
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airlinesearch(iata: str=None, name: str=None, icao: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of airlines for a given country, identifiers or name search pattern. Multiple arguments can be specified to limit the search.
		Either one of the arguments must be specified."
    iata: The IATA code of the airline
        name: The name or part of the name of the airline, case insensitive
        icao: The ICAO code of the airline
        country: The ISO 3166-1 alpha-2 code country code to request
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/airline/search"
    querystring = {}
    if iata:
        querystring['iata'] = iata
    if name:
        querystring['name'] = name
    if icao:
        querystring['icao'] = icao
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aircraftsearch(model: str=None, airline_ident: str=None, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of plane registrations for a given search criteria.
		Can search by airline ident, country, model
		If multiple arguments are provided, they will be chained as AND
		Returns only currently active aircraft"
    model: Model ICAO code
        airline_ident: The ident of the airline, as returned by /airline/search
        country: Two letter country code
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/aircraft/search"
    querystring = {}
    if model:
        querystring['model'] = model
    if airline_ident:
        querystring['airline_ident'] = airline_ident
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airportsearch(timezone: str=None, country: str=None, bbox: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of airports for a given country, bbox or timezone. Multiple arguments can be specified to limit the search.
		Either one of country, bbox or timezone must be specified."
    timezone: The timezone of the airport (e.g. Europe/Berlin)
        country: The ISO 3166-1 alpha-2 code country code to request
        bbox: A bbox (min Longitude , min Latitude , max Longitude , max Latitude), will restrict results to airports within that box
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/airport/search"
    querystring = {}
    if timezone:
        querystring['timezone'] = timezone
    if country:
        querystring['country'] = country
    if bbox:
        querystring['bbox'] = bbox
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airportinfo(icao: str=None, localid: str=None, iata: str=None, ident: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns detailed airport information for airports identified by the provided identifiers. In most cases the list should contain only one entry.
		However, due to the nature of the data, it is possible that multiple entries are returned.
		If searched by ident, only one airport is returned.
		One of ident, iata, icao, localID must be provided."
    icao: The ICAO code of the airport, must have a length of 4
        localid: Local identifiers of the airport, currently mostly available for the US & CA (FAA-ID). Max length is 6
        iata: The IATA code of the airport, must have a length of 3
        ident: The ident of the airport, as returned by /airport/search
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/airport/info"
    querystring = {}
    if icao:
        querystring['icao'] = icao
    if localid:
        querystring['localID'] = localid
    if iata:
        querystring['iata'] = iata
    if ident:
        querystring['ident'] = ident
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airlineaircrafts(ident: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of plane registrations for a given airline. Can search by ident only.
		Only returns currently active aircraft"
    ident: The ident of the airline, as returned by /airline/search
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/airline/aircrafts"
    querystring = {'ident': ident, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flightstatistics(flnr: str, aptfrom: str=None, aptto: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns statistics for a flight number with one entry per route flown. Statistics are updated daily.
		The query can be filtered by route.
		Statistics are always as of today, please contact us if you require historical statistics."
    flnr: The flight number to request
        aptfrom: Departure airport ident
        aptto: Arrival airport ident
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/flight/statistics"
    querystring = {'flnr': flnr, }
    if aptfrom:
        querystring['aptFrom'] = aptfrom
    if aptto:
        querystring['aptTo'] = aptto
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airlinestatistics(icao: str=None, ident: str=None, iata: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns detailed airline statistics for airlines identified by the provided identifiers. In most cases the list should contain only one entry, unless the ident is ambiguous.
		If searched by ident, only one airline is returned.
		One of ident, iata, icao must be provided.
		Statistics are calculated overnight and represent the current day's data."
    icao: The ICAO code of the airline, must have a length of 4
        ident: The ident of the airline, as returned by /airline/search
        iata: The IATA code of the airline, must have a length of 3
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/airline/statistics"
    querystring = {}
    if icao:
        querystring['icao'] = icao
    if ident:
        querystring['ident'] = ident
    if iata:
        querystring['iata'] = iata
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flightinfo(flnr: str, date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the info for a flight on a specified date, or the current flight if date is omitted. Will return a 400 if the date requested is outside the subscription level"
    flnr: The flight number to request
        date: The date, if omitted the current flight will be returned
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/flight/info"
    querystring = {'flnr': flnr, }
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airportmetar(ident: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the most recent METAR for an airport specified by the given ident.
		Includes a parsed version of the METAR."
    ident: The ident of the airport (e.g. as returned by /airport/search)
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/airport/metar"
    querystring = {'ident': ident, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airportdelaydailystatistics(isdeparture: bool, dt: str, ident: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns detailed daily delay statistics for an airport identified by the provided identifiers. If searched by ident, only one airport is returned.
		One of ident, iata, icao, localID must be provided. Additionally, `dt` and `isDeparture` are required.
		Statistics are calculated overnight and represent the data for the specific date."
    isdeparture: Flag indicating if the desired statistics are for departures (true) or arrivals (false)
        dt: The date for which delay statistics are desired, formatted as "YYYY-MM-DD"
        ident: The ident of the airport, as returned by /airport/search
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/airport/delays_by_day"
    querystring = {'isDeparture': isdeparture, 'dt': dt, }
    if ident:
        querystring['ident'] = ident
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airportflights(direction: str, ident: str, time: str, counterpart_ident: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of flights for an airport, direction and time. The list is sorted ascending by scheduled departure time for direction=="departure" and sorted descending by scheduled arrival time for direction=="arrival".
		The next departure/arrival time is returned for pagination.
		The dates returned for each flight represent the scheduled departure date in local time and can be used to query flight data via /flight/info."
    direction: Direction, one of "arrival", "departure"
        ident: Ident of the airport to request. Ident is the unique identifier as returned by /airport/search or /airport/info endpoints.
        time: Timestamp in UTC. If direction is "arrival", show flights with scheduled arrival time prior to this timestamp. If direction is "departure", show flights with scheduled departure time after this timestamp. Expected in any valid format, such as 2022-01-01T12:00:00.000Z. Maximum range of flights returned is 30 days.
        counterpart_ident: Ident of the counterpart airport (destination if direction=arrival, otherwise arrival). Used to query connections on a defined route. Ident is the unique identifier as returned by /airport/search or /airport/info endpoints.
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/airport/flights"
    querystring = {'direction': direction, 'ident': ident, 'time': time, }
    if counterpart_ident:
        querystring['counterpart_ident'] = counterpart_ident
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aircraftinfo(reg: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns detailed aircraft information for an aircraft identified by registration.
		Only works for currently active aircraft"
    reg: The registration of the aircraft, as returned by /airline/aircraft or /aircraft/search
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/aircraft/info"
    querystring = {'reg': reg, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airlineflights(time: str, ident: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of flights for an airline given a start time. The list is sorted ascending by scheduled departure time.
		The next departure/ time is returned for pagination (use start times to paginate).
		The dates returned for each flight represent the scheduled departure date in local time and can be used to query flight data via /flight/info."
    time: Timestamp in UTC. Shows flights with scheduled departure time after this timestamp. Expected in any valid format, such as 2022-01-01T12:00:00.000Z. Maximum range of flights returned is 30 days.
        ident: Ident of the airline to request. Ident is the unique identifier as returned by /airline/search or /airline/info endpoints.
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/airline/flights"
    querystring = {'time': time, 'ident': ident, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airlineinfo(icao: str=None, name: str=None, iata: str=None, ident: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns detailed airline information for airlines identified by the provided identifiers. In most cases the list should contain only one entry.
		However, due to the nature of the data, it is possible that multiple entries are returned.
		If searched by ident, only one airline is returned.
		One of ident, iata, icao, name must be provided."
    icao: The ICAO code of the airline, must have a length of 4
        name: The name of the airline. Not case sensitive
        iata: The IATA code of the airline, must have a length of 3
        ident: The ident of the airline, as returned by /airline/search
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/airline/info"
    querystring = {}
    if icao:
        querystring['icao'] = icao
    if name:
        querystring['name'] = name
    if iata:
        querystring['iata'] = iata
    if ident:
        querystring['ident'] = ident
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airportstatistics(ident: str=None, localid: str=None, iata: str=None, icao: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns detailed airport statistics for airports identified by the provided identifiers. In most cases the list should contain only one entry, unless the ident is ambiguous.
		If searched by ident, only one airport is returned.
		One of ident, iata, icao, localID must be provided.
		Statistics are calculated overnight and represent the current day's data."
    ident: The ident of the airport, as returned by /airport/search
        localid: Local identifiers of the airport, currently mostly available for the US & CA (FAA-ID). Max length is 6
        iata: The IATA code of the airport, must have a length of 3
        icao: The ICAO code of the airport, must have a length of 4
        
    """
    url = f"https://flightera-flight-data.p.rapidapi.com/airport/statistics"
    querystring = {}
    if ident:
        querystring['ident'] = ident
    if localid:
        querystring['localID'] = localid
    if iata:
        querystring['iata'] = iata
    if icao:
        querystring['icao'] = icao
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flightera-flight-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

