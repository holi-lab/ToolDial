import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_1_3_get_info_day(country_code: str, date: str, configuration: str='Federal holidays', profile_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information about a specific day."
    country_code: The ISO country code (2 letters).  See <a href=https://api.workingdays.org/1.2/api-countries.php>available countries & configurations</a>
        date: The date to analyze (YYYY-MM-DD)
        configuration: The name of the preset configuration to be used. See <a href=https://api.workingdays.org/1.2/api-countries.php>available countries & configurations</a>
        
    """
    url = f"https://working-days.p.rapidapi.com/1.3/get_info_day"
    querystring = {'country_code': country_code, 'date': date, }
    if configuration:
        querystring['configuration'] = configuration
    if profile_id:
        querystring['profile_id'] = profile_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "working-days.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_3_delete_custom_period(start_date: str, profile_id: str, end_date: str='2013-01-07', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Removing a previously defined custom period"
    start_date: The start date of the custom period (YYYY-MM-DD)
        profile_id: The ID of the calendar we are customizing.
        end_date: The end date of the custom period (YYYY-MM-DD) If omitted, end date will be equal to start date (one day custom period)
        
    """
    url = f"https://working-days.p.rapidapi.com/1.3/delete_custom_period"
    querystring = {'start_date': start_date, 'profile_id': profile_id, }
    if end_date:
        querystring['end_date'] = end_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "working-days.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_3_analyse(start_date: str, end_date: str, country_code: str, end_time: str='18:15', start_time: str='09:14', profile_id: str=None, configuration: str='Federal holidays', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Analyse a period (you provide a start_date and an end_date over a given calendar, we respond with the period analysis)"
    start_date: The start date (YYYY-MM-DD)
        end_date: The end date (YYYY-MM-DD)
        country_code: The ISO country code (2 letters).  See <a href=https://api.workingdays.org/api-countries >available countries & configurations</a>
        end_time: The end date's time (24 hours format, like 09:00 or 15:00, but not 3pm).If omitted, default value is 23:59.
        start_time: The start date's time (24 hours format, like 09:00 or 15:00, but not 3pm).If omitted, default value is 00:00
        configuration: The name of the preset configuration to be used. See <a href=https://api.workingdays.org/api-countries >available countries & configurations</a>
        
    """
    url = f"https://working-days.p.rapidapi.com/1.3/analyse"
    querystring = {'start_date': start_date, 'end_date': end_date, 'country_code': country_code, }
    if end_time:
        querystring['end_time'] = end_time
    if start_time:
        querystring['start_time'] = start_time
    if profile_id:
        querystring['profile_id'] = profile_id
    if configuration:
        querystring['configuration'] = configuration
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "working-days.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_3_add_working_hours(start_date: str, country_code: str, start_time: str, increment_time: str='1815', configuration: str='Federal holidays', profile_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Add an amount of working time to a given start date/time"
    start_date: The start date (YYYY-MM-DD)
        country_code: The ISO country code (2 letters).  See <a href=https://api.workingdays.org/api-countries >available countries & configurations</a>
        start_time: The start time in a 24 hours format with leading zeros.
        increment_time: The amount of working time to be added (or removed) to the start date time. Format H:i. This amount cannot exceed 5000 hours. For example, to add one hour&#58; 1&#58;00. To add 30 hours and 15 minutes&#58; 30:15.  To remove 45 minutes&#58; -0:45
        configuration: The name of the preset configuration to be used. See <a href=https://api.workingdays.org/api-countries >available countries & configurations</a>
        
    """
    url = f"https://working-days.p.rapidapi.com/1.3/add_working_hours"
    querystring = {'start_date': start_date, 'country_code': country_code, 'start_time': start_time, }
    if increment_time:
        querystring['increment_time'] = increment_time
    if configuration:
        querystring['configuration'] = configuration
    if profile_id:
        querystring['profile_id'] = profile_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "working-days.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_3_list_non_working_days(start_date: str, end_date: str, country_code: str, configuration: str='Federal holidays', profile_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the non working days (weekend days, public holidays and custom dates) between two dates in chronlogical order."
    start_date: The start date (YYYY-MM-DD)
        end_date: The end date (YYYY-MM-DD)
        country_code: The ISO country code (2 letters).  See <a href=https://api.workingdays.org/api-countries >available countries & configurations</a>
        configuration: The name of the preset configuration to be used. See <a href=https://api.workingdays.org/api-countries >available countries & configurations</a>
        
    """
    url = f"https://working-days.p.rapidapi.com/1.3/list_non_working_days"
    querystring = {'start_date': start_date, 'end_date': end_date, 'country_code': country_code, }
    if configuration:
        querystring['configuration'] = configuration
    if profile_id:
        querystring['profile_id'] = profile_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "working-days.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_3_define_custom_period(description: str, profile_id: str, start_date: str, color: str='orange', start_morning: bool=None, end_afternoon: bool=None, end_date: str='2013-01-07', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Defining a custom period (typically days of vacations of an employee or days of annual closure of a company) can be done from the working days websites user interface but can also be done programmatically from the API. A typical use case would be to retrieve the vacations of the workers out of the human ressources software and insert them into a custom calendar, then being able to query available working days."
    description: A textual description of the custom period (up to 30 caracters)
        profile_id: The ID of the calendar we are customizing.
        start_date: The start date of the custom period (YYYY-MM-DD)
        color: The color of custom period when displayed on the calendar. Colors can be useful in order to classify custom periods. Default value is orange.
        start_morning: Does the custom period starts in the morning of the start_date? Default value is true. (This parameter can be used if you want to create half day custom periods.)
        end_afternoon: Does the custom period ends in the afternoon of the end_date? Default value is true. (This parameter can be used if you want to create half day custom periods.)
        end_date: The end date of the custom period (YYYY-MM-DD) If omitted, end date will be equal to start date (one day custom period)
        
    """
    url = f"https://working-days.p.rapidapi.com/1.3/define_custom_period"
    querystring = {'description': description, 'profile_id': profile_id, 'start_date': start_date, }
    if color:
        querystring['color'] = color
    if start_morning:
        querystring['start_morning'] = start_morning
    if end_afternoon:
        querystring['end_afternoon'] = end_afternoon
    if end_date:
        querystring['end_date'] = end_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "working-days.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_3_add_working_days(country_code: str, start_date: str, increment: int, include_start: bool=None, configuration: str='Federal holidays', profile_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Add (or remove) any number of working days to a date."
    country_code: The ISO country code (2 letters).  See <a href=https://api.workingdays.org/api-countries >available countries & configurations</a>
        start_date: The start date (YYYY-MM-DD)
        increment: The number of working days you want to add to your start date (positive or negative integer but not zero)
        include_start: Should the count include the start_date? Default value is true. If you set include_start to false ("false" or "0"), the count will start at the next working day (or previous working day, if increment is negative)
        configuration: The name of the preset configuration to be used. See <a href=https://api.workingdays.org/api-countries.php>available countries & configurations</a>
        
    """
    url = f"https://working-days.p.rapidapi.com/1.3/add_working_days"
    querystring = {'country_code': country_code, 'start_date': start_date, 'increment': increment, }
    if include_start:
        querystring['include_start'] = include_start
    if configuration:
        querystring['configuration'] = configuration
    if profile_id:
        querystring['profile_id'] = profile_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "working-days.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

