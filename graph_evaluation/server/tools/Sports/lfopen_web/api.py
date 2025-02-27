import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_results_workout(fromdate: str, todate: str, timezone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves workout progress of User between the specified dates"
    fromdate: Format : MM/dd/YYYY
        todate: Format : MM/dd/YYYY
        timezone: Format : GMT Sign Hours : Minutes (e.g. "GMT-8:00") or GMT Sign Hours Minutes (e.g. "GMT-0800") or GMT Sign Hours (e.g. "GMT-8").
        
    """
    url = f"https://life-fitness-lfconnect-website1.p.rapidapi.com/workoutresults/get_results_workout"
    querystring = {'fromDate': fromdate, 'toDate': todate, 'timezone': timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "life-fitness-lfconnect-website1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_results_daily(fromdate: str, todate: str, timezone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves Daily Progress of User between the specified dates"
    fromdate: Format : MM/dd/YYYY
        todate: Format : MM/dd/YYYY
        timezone: Format : GMT Sign Hours : Minutes (e.g. "GMT-8:00") or GMT Sign Hours Minutes (e.g. "GMT-0800") or GMT Sign Hours (e.g. "GMT-8").
        
    """
    url = f"https://life-fitness-lfconnect-website1.p.rapidapi.com/workoutresults/get_results_daily"
    querystring = {'fromDate': fromdate, 'toDate': todate, 'timezone': timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "life-fitness-lfconnect-website1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_lifefitness_results(fromdate: str, todate: str, timezone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    fromdate: Format : MM/dd/YYYY
        todate: Format : MM/dd/YYYY
        timezone: Format : GMT Sign Hours : Minutes (e.g. "GMT-8:00") or GMT Sign Hours Minutes (e.g. "GMT-0800") or GMT Sign Hours (e.g. "GMT-8").
        
    """
    url = f"https://life-fitness-lfconnect-website1.p.rapidapi.com/workoutresults/get_lifefitness_results"
    querystring = {'fromDate': fromdate, 'toDate': todate, 'timezone': timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "life-fitness-lfconnect-website1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_account_details(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the user account details based on the User access token passed."
    
    """
    url = f"https://life-fitness-lfconnect-website1.p.rapidapi.com/user"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "life-fitness-lfconnect-website1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_photo(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the Profile photo of the LFConnect User related to the Access Token passed."
    
    """
    url = f"https://life-fitness-lfconnect-website1.p.rapidapi.com/user/getphoto"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "life-fitness-lfconnect-website1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_equipment_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retreives all the equipments available.  This service retrieves the static values of equipment Ids and equipments. These values can be provided as an input to get the User workouts using /get_workoutlist service.  The workout results retrieval services holds these values to identify the equipment that the workout has been completed"
    
    """
    url = f"https://life-fitness-lfconnect-website1.p.rapidapi.com/workoutpreset/get_equipment_list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "life-fitness-lfconnect-website1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_gps_results(fromdate: str, todate: str, timezone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves Workout results posted by User through GPS between the specified dates"
    fromdate: Format : MM/dd/YYYY
        todate: Format : MM/dd/YYYY
        timezone: Format : GMT Sign Hours : Minutes (e.g. "GMT-8:00") or GMT Sign Hours Minutes (e.g. "GMT-0800") or GMT Sign Hours (e.g. "GMT-8").
        
    """
    url = f"https://life-fitness-lfconnect-website1.p.rapidapi.com/workoutresults/get_gps_results"
    querystring = {'fromDate': fromdate, 'toDate': todate, 'timezone': timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "life-fitness-lfconnect-website1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_manualstrength_results(fromdate: str, todate: str, timezone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves Manual Strength workout results of User between the specifieddates"
    fromdate: Format : MM/dd/YYYY
        todate: Format : MM/dd/YYYY
        timezone: Format : GMT Sign Hours : Minutes (e.g. "GMT-8:00") or GMT Sign Hours Minutes (e.g. "GMT-0800") or GMT Sign Hours (e.g. "GMT-8").
        
    """
    url = f"https://life-fitness-lfconnect-website1.p.rapidapi.com/workoutresults/get_manualstrength_results"
    querystring = {'fromDate': fromdate, 'toDate': todate, 'timezone': timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "life-fitness-lfconnect-website1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_summary_results(fromdate: str, todate: str, timezone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Read current progress information.  This service provides the progress information like calories,duration,workout span days and weights lifted per Day,Week,Month and Year. The currDate parameter value passed is considered as the current day till which the progress information is seeked."
    fromdate: Format : MM/dd/YYYY
        todate: Format : MM/dd/YYYY
        timezone: Format : GMT Sign Hours : Minutes (e.g. "GMT-8:00") or GMT Sign Hours Minutes (e.g. "GMT-0800") or GMT Sign Hours (e.g. "GMT-8").
        
    """
    url = f"https://life-fitness-lfconnect-website1.p.rapidapi.com/workoutresults/get_summary_results"
    querystring = {'fromDate': fromdate, 'toDate': todate, 'timezone': timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "life-fitness-lfconnect-website1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_workoutlist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The user creates custom workouts giving specific goals and parameters such as time/calories/distance. This service returns all the workout presets created by the user. The user selects any one of the preset in the Lifefitness equipment before starting a workout  If the equipmentIds are not mentioned, then this returns all the workouts available. This details provides the users preference of workouts and might help in suggesting similar workout regimens.  To provide the value for equipmentIds input, the service xx can be used to get the list of predefined equipmentIds and the desired value can be provided."
    
    """
    url = f"https://life-fitness-lfconnect-website1.p.rapidapi.com/workoutpreset/get_workoutlist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "life-fitness-lfconnect-website1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_manual_cardio_results(fromdate: str, todate: str, timezone: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves Manual Cardio workout results of User between the specified dates"
    fromdate: Format : MM/dd/YYYY
        todate: Format : MM/dd/YYYY
        timezone: Format : GMT Sign Hours : Minutes (e.g. "GMT-8:00") or GMT Sign Hours Minutes (e.g. "GMT-0800") or GMT Sign Hours (e.g. "GMT-8").
        
    """
    url = f"https://life-fitness-lfconnect-website1.p.rapidapi.com/workoutresults/get_manualcardio_results"
    querystring = {'fromDate': fromdate, 'toDate': todate, 'timezone': timezone, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "life-fitness-lfconnect-website1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

