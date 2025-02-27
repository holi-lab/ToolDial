import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def listoffinishingstatusforaspecificseason(year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To fetch the list of finishing status of a specific season(year).
		"
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def driverstandingsafterarace(year: str, round: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To fetch the driver standings after a specific race(round) in a season(year), use this endpoint."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/{round}/driverStandings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listofalldrivers(q: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To obtain a list of all drivers.
		
		Driver lists can be refined by adding one or more of the following criteria:
		
		/circuits/<circuitId>
		/constructors/<constructorId>
		/drivers/<driverId>
		/grid/<position>
		/results/<position>
		/fastest/<rank>
		/status/<statusId>
		
		For example, to list all drivers who have driven for a specific constructor at a particular circuit:
		
		http://ergast.com/api/f1/constructors/mclaren/circuits/monza/drivers
		
		Alternatively, to list the drivers who have achieved a particular final position in the championship:
		
		http://ergast.com/api/f1/driverStandings/1/drivers
		
		Drivers who participated in the 2014 season onwards have a permanent driver number."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/drivers"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def circuitinformation(circuitid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each constructor listed in the response is identified by a unique circuitId which is used to identify the constructor throughout the API. To obtain information about a particular circuit append the circuitId (name of the circuit)
		
		This endpoint is to obtain the information of a particular circuit based on the circuitId (circuit name)"
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/circuits/{circuitid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listofalldriverswithinaraceinayear(round: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is to obtain a list of all drivers in a particular race(round) of a season(year)"
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/{round}/drivers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listofalldriverswithinayear(year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is to obtain a list of all drivers in a particular season (year)"
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/drivers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listofallcircuits(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is to obtain a list of all constructors.
		
		
		Circuit lists can be refined by adding one or more of the following criteria:
		
		/constructors/<constructorId>
		/drivers/<driverId>
		/grid/<position>
		/results/<position>
		/fastest/<rank>
		/status/<statusId>
		
		For example, to list all circuits at which a specific driver has driven for a particular constructor:
		
		http://ergast.com/api/f1/drivers/alonso/constructors/mclaren/circuits"
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/circuits.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def driverinformation(driverid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each driver listed in the response is identified by a unique driverId which is used to identify the driver throughout the API. To obtain information about a particular driver append the driverId
		
		This endpoint is to obtain the information of a particular driver based on the driverid (driver name)"
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/drivers/{driverid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listofallcircuitswithinayear(year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/circuits"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listofallcircuitswithinaraceinayear(round: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/{round}/circuits"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listofallconstructors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is to obtain a list of all constructors.
		
		Constructor lists can be refined by adding one or more of the following criteria:
		
		/circuits/<circuitId>
		/constructors/<constructorId>
		/drivers/<driverId>
		/grid/<position>
		/results/<position>
		/fastest/<rank>
		/status/<statusId>
		
		For example, to list all constructors a specific driver has driven for at a particular circuit:
		
		http://ergast.com/api/f1/drivers/alonso/circuits/monza/constructors
		
		Alternatively, to list all the constructors who have achieved a particular final position in the championship:
		
		http://ergast.com/api/f1/constructorStandings/1/constructors"
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/constructors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasonslist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives the seasons currently supported by the API.
		
		
		Season lists can be refined by adding one or more of the following criteria:
		
		/circuits/<circuitId>
		/constructors/<constructorId>
		/drivers/<driverId>
		/grid/<position>
		/results/<position>
		/fastest/<rank>
		/status/<statusId>
		
		For example, to list all seasons where a specific driver has driven for a particular constructor:
		
		http://ergast.com/api/f1/drivers/alonso/constructors/renault/seasons
		
		Alternatively, to list the seasons where a specific driver or constructor has achieved a particular final position in the championship:
		
		http://ergast.com/api/f1/drivers/alonso/driverStandings/1/seasons
		http://ergast.com/api/f1/constructors/renault/constructorStandings/1/seasons"
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/seasons"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def informationforaspecificpitstop(year: str, pitstopnumber: str, round: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint requires the season(year), race(round) and pit stop number(pitstopnumber) to be specified."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/{round}/pitstops/{pitstopnumber}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def allwinnersofdrivers_championships(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To fetch all the winners information of drivers."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/driverStandings/1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasonenddriverstandings(year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Driver Standings at the end of the season(year)."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/driverStandings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listofallfinishingstatuscodes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Every driver has a finishing status like 'finished', disqualified', 'accident', '+1 lap', '+2 lap', etc. This endpoint will give the list of all such statuses. Example request and response attached"
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def infoaboutaspecificrace(year: str, round: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To obtain the schedule of races for a specific race(round) in a season(year), use this endpoint."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/{round}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listoffinishingstatusforaspecificroundinaseason(round: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To fetch the list of finishing status of a specific race(round) in a season(year)."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/{round}/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mostrecentraceresult(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Results from the most recent race can be obtained using this endpoint."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/current/last/results"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scheduleofracesforaseason(year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To obtain the schedule of races for a specific season use this endpoint required season (year)."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listofallconstructorswithinayear(year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is to obtain a list of all constructors in a particular season (year)"
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/constructors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def constructorinformation(constructorid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each constructor listed in the response is identified by a unique constructorId which is used to identify the constructor throughout the API. To obtain information about a particular constructor append the constructorId (name of the constructor)
		
		This endpoint is to obtain the information of a particular constructor based on the constructorId (constructor name)"
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/constructors/{constructorid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def constructorstandingsbyspecifyingtheconstructor(constructorid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Constructor standings by giving in the constructorid(name of the constructor)."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/constructors/{constructorid}/constructorStandings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currentdrivers_standing(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current driver standings can always be obtained using this endpoint."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/current/driverStandings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pitstopdataforarace(year: str, round: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint requires the season(year) and race(round) to be specified.
		"
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/{round}/pitstops"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listofallconstructorswithinaraceinayear(year: str, round: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is to obtain a list of all constructors in a particular race(round) of a season(year)"
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/{round}/constructors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seasonendconstructorstanding(year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Constructor Standings at the end of the season(year)."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/constructorStandings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def constructorstandingsafterarace(year: str, round: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To list the constructor standings after a specific race (round) in a season(year), use this endpoint.
		
		"
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/{round}/constructorStandings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def raceresult(round: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint gives the result for a specific race (round) in a season(year).
		
		
		If the results for the specified race are not yet available the RaceTable element in the response will be empty.
		
		"
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/{round}/results"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scheduleofthecurrentseason(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To obtain the schedule of races for the current season, use this endpoint."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/current"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def qualifyingresults(year: str, round: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/{round}/qualifying"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def laptime(lapnumber: str, round: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint requires the season(year), race(round) and lap number(lapnumber) to be specified."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/{year}/{round}/laps/{lapnumber}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currentconstructor_sstanding(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Current constructor standings can always be obtained using this endpoint."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/current/constructorStandings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def driverstandingsbyspecifyingthedriver(driverid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Driver standings by giving in the driverid(name of the driver)."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/drivers/{driverid}/driverStandings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def allwinnersofconstructors_championships(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To fetch all the winners information of constructors."
    
    """
    url = f"https://formula-1-fixed.p.rapidapi.com/api/f1/constructorStandings/1"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "formula-1-fixed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

