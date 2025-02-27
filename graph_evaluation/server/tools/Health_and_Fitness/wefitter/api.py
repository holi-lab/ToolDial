import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def team_read(public_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single team"
    
    """
    url = f"https://wefitter2.p.rapidapi.com/team/{public_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def team_list(offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available teams"
    offset: The initial index from which to return the results.
        limit: Number of results to return per page.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/team/"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_list(ordering: str=None, limit: int=None, offset: int=None, search: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all available profiles"
    ordering: Which field to use when ordering the results.
        limit: Number of results to return per page.
        offset: The initial index from which to return the results.
        search: A search term.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/"
    querystring = {}
    if ordering:
        querystring['ordering'] = ordering
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if search:
        querystring['search'] = search
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_daily_summary_list(profile_public_id: str, date_start: str=None, date_end: str=None, cursor: str=None, deduplicate: bool=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides summarized data per day."
    date_start: Lower bound of date or timestamp.
        date_end: Upper bound of date or timestamp.
        cursor: The pagination cursor value.
        deduplicate: If set to true, duplicates will be aggregated.
        page_size: Number of results to return per page.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/daily_summary/"
    querystring = {}
    if date_start:
        querystring['date_start'] = date_start
    if date_end:
        querystring['date_end'] = date_end
    if cursor:
        querystring['cursor'] = cursor
    if deduplicate:
        querystring['deduplicate'] = deduplicate
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_bmi_list(profile_public_id: str, date_end: str=None, date_start: str=None, page_size: int=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Presents the Body-Mass-Index over time."
    date_end: Upper bound of date or timestamp.
        date_start: Lower bound of date or timestamp.
        page_size: Number of results to return per page.
        cursor: The pagination cursor value.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/bmi/"
    querystring = {}
    if date_end:
        querystring['date_end'] = date_end
    if date_start:
        querystring['date_start'] = date_start
    if page_size:
        querystring['page_size'] = page_size
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def challenge_periods(public_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all periods for the challenge.
		This endpoint returns a list of objects.
		"
    
    """
    url = f"https://wefitter2.p.rapidapi.com/challenge/{public_id}/periods/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_stress_summary_list(profile_public_id: str, cursor: str=None, page_size: int=None, date_start: str=None, date_end: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Presents stress summaries."
    cursor: The pagination cursor value.
        page_size: Number of results to return per page.
        date_start: Lower bound of date or timestamp.
        date_end: Upper bound of date or timestamp.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/stress_summary/"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if page_size:
        querystring['page_size'] = page_size
    if date_start:
        querystring['date_start'] = date_start
    if date_end:
        querystring['date_end'] = date_end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_connections_read(public_id: str, redirect: str=None, redirect_on_error: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists the connections available for a profile and their "state".
		This endpoint returns a list of objects.
		"
    redirect: After connecting redirect back to this url.

Note: has to be a valid http url. For deeplinks (to apps) first redirect to your own backend and then deeplink from there.

Note: the user is only redirected on a successfull connection, otherwise an error screen is shown. (see redirect_on_error to change this)

Note: if this parameter is omitted, then user will see a generic success screen.
        redirect_on_error: If enabled, then user will also be redirected to the redirect url if an error occurs

Note: a query parameter 'error' is added to the redirect url. Possible values are access_denied, unknown. New values may be added at anytime

Note: for backwards compatibility the default is false, but this will be changed in the next breaking release
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{public_id}/connections/"
    querystring = {}
    if redirect:
        querystring['redirect'] = redirect
    if redirect_on_error:
        querystring['redirect_on_error'] = redirect_on_error
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_challenges(public_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This docstring is used by Django Rest Framework"
    
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{public_id}/challenges/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_height_list(profile_public_id: str, date_end: str=None, date_start: str=None, page_size: int=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists changes in height for the profile."
    date_end: Upper bound of date or timestamp.
        date_start: Lower bound of date or timestamp.
        page_size: Number of results to return per page.
        cursor: The pagination cursor value.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/height/"
    querystring = {}
    if date_end:
        querystring['date_end'] = date_end
    if date_start:
        querystring['date_start'] = date_start
    if page_size:
        querystring['page_size'] = page_size
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_heartrate_sample_list(profile_public_id: str, date_start: str=None, page_size: int=None, cursor: str=None, date_end: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Heart Rate measured at a single point in time."
    date_start: Lower bound of date or timestamp.
        page_size: Number of results to return per page.
        cursor: The pagination cursor value.
        date_end: Upper bound of date or timestamp.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/heartrate_sample/"
    querystring = {}
    if date_start:
        querystring['date_start'] = date_start
    if page_size:
        querystring['page_size'] = page_size
    if cursor:
        querystring['cursor'] = cursor
    if date_end:
        querystring['date_end'] = date_end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_event_list(profile_public_id: str, date_end: str=None, date_start: str=None, cursor: str=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists events, generic actions for which points have been awarded."
    date_end: Upper bound of date or timestamp.
        date_start: Lower bound of date or timestamp.
        cursor: The pagination cursor value.
        page_size: Number of results to return per page.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/event/"
    querystring = {}
    if date_end:
        querystring['date_end'] = date_end
    if date_start:
        querystring['date_start'] = date_start
    if cursor:
        querystring['cursor'] = cursor
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_biometric_measurement_list(measurement_type: str, profile_public_id: str, date_end: str=None, date_start: str=None, page_size: int=None, cursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List changes in biometric measurements for the profile."
    date_end: Upper bound of date or timestamp.
        date_start: Lower bound of date or timestamp.
        page_size: Number of results to return per page.
        cursor: The pagination cursor value.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/biometric_measurement/"
    querystring = {'measurement_type': measurement_type, }
    if date_end:
        querystring['date_end'] = date_end
    if date_start:
        querystring['date_start'] = date_start
    if page_size:
        querystring['page_size'] = page_size
    if cursor:
        querystring['cursor'] = cursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_read(public_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a single profile"
    
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{public_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_ecg_samples_list(profile_public_id: str, date_end: str=None, page_size: int=None, cursor: str=None, date_start: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "
		To get electrocardiogram (ECG) recordings, the endpoint `/ecg_samples/` can be used. 
		This endpoint return an ECG recording as an array of tuples `[(time, value)]`. Other information that is returned is the duration
		of the recording, sample frequency and sample count.
		"
    date_end: Upper bound of date or timestamp.
        page_size: Number of results to return per page.
        cursor: The pagination cursor value.
        date_start: Lower bound of date or timestamp.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/ecg_samples/"
    querystring = {}
    if date_end:
        querystring['date_end'] = date_end
    if page_size:
        querystring['page_size'] = page_size
    if cursor:
        querystring['cursor'] = cursor
    if date_start:
        querystring['date_start'] = date_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_challenge_read(public_id: str, date_start: str=None, recent: int=None, date_end: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a list of challenges where the profile can participate and is already participating in.
		
		This endpoint returns a list of objects.
		"
    date_start: Only return challenges that are active after this time

Note: this only affects challenges that are not joined.

Defaults to now
        recent: Show challenges that have not ended in the last N weeks
        date_end: Only return challenges that are active before this time

Note: this only affects challenges that are not joined.

Defaults to now
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{public_id}/challenge/"
    querystring = {}
    if date_start:
        querystring['date_start'] = date_start
    if recent:
        querystring['recent'] = recent
    if date_end:
        querystring['date_end'] = date_end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_heartrate_summary_list(profile_public_id: str, cursor: str=None, date_end: str=None, date_start: str=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Presents heart rate information."
    cursor: The pagination cursor value.
        date_end: Upper bound of date or timestamp.
        date_start: Lower bound of date or timestamp.
        page_size: Number of results to return per page.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/heartrate_summary/"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if date_end:
        querystring['date_end'] = date_end
    if date_start:
        querystring['date_start'] = date_start
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_stress_samples_list(profile_public_id: str, page_size: int=None, cursor: str=None, date_end: str=None, date_start: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Presents stress samples."
    page_size: Number of results to return per page.
        cursor: The pagination cursor value.
        date_end: Upper bound of date or timestamp.
        date_start: Lower bound of date or timestamp.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/stress_samples/"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if cursor:
        querystring['cursor'] = cursor
    if date_end:
        querystring['date_end'] = date_end
    if date_start:
        querystring['date_start'] = date_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_daily_detail_list(profile_public_id: str, page_size: int=None, date_start: str=None, cursor: str=None, date_end: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Intraday data measured at a single point in time."
    page_size: Number of results to return per page.
        date_start: Lower bound of date or timestamp.
        cursor: The pagination cursor value.
        date_end: Upper bound of date or timestamp.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/daily_detail/"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if date_start:
        querystring['date_start'] = date_start
    if cursor:
        querystring['cursor'] = cursor
    if date_end:
        querystring['date_end'] = date_end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_workout_list(profile_public_id: str, page_size: int=None, deduplicate: bool=None, date_start: str=None, cursor: str=None, date_end: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists workouts associated with the profile. Used to be known as Activity."
    page_size: Number of results to return per page.
        deduplicate: If set to true, duplicates will be aggregated.
        date_start: Lower bound of date or timestamp.
        cursor: The pagination cursor value.
        date_end: Upper bound of date or timestamp.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/workout/"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if deduplicate:
        querystring['deduplicate'] = deduplicate
    if date_start:
        querystring['date_start'] = date_start
    if cursor:
        querystring['cursor'] = cursor
    if date_end:
        querystring['date_end'] = date_end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_connection_widget(public_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Presents an HTML connection widget."
    
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{public_id}/connection_widget/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_totals_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Total values for calories, distance, steps, points and activity duration over all profiles in the app."
    
    """
    url = f"https://wefitter2.p.rapidapi.com/app/totals/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_activity_segment_list(profile_public_id: str, cursor: str=None, page_size: int=None, date_start: str=None, date_end: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists segments of activities and small non-workout-like movements."
    cursor: The pagination cursor value.
        page_size: Number of results to return per page.
        date_start: Lower bound of date or timestamp.
        date_end: Upper bound of date or timestamp.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/activity_segment/"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if page_size:
        querystring['page_size'] = page_size
    if date_start:
        querystring['date_start'] = date_start
    if date_end:
        querystring['date_end'] = date_end
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_weight_list(profile_public_id: str, cursor: str=None, date_end: str=None, page_size: int=None, date_start: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists changes in weight for the profile."
    cursor: The pagination cursor value.
        date_end: Upper bound of date or timestamp.
        page_size: Number of results to return per page.
        date_start: Lower bound of date or timestamp.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/weight/"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if date_end:
        querystring['date_end'] = date_end
    if page_size:
        querystring['page_size'] = page_size
    if date_start:
        querystring['date_start'] = date_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_biometric_list(profile_public_id: str, page_size: int=None, date_end: str=None, cursor: str=None, date_start: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists changes in weight, height and bmi for the profile."
    page_size: Number of results to return per page.
        date_end: Upper bound of date or timestamp.
        cursor: The pagination cursor value.
        date_start: Lower bound of date or timestamp.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/biometric/"
    querystring = {}
    if page_size:
        querystring['page_size'] = page_size
    if date_end:
        querystring['date_end'] = date_end
    if cursor:
        querystring['cursor'] = cursor
    if date_start:
        querystring['date_start'] = date_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_sleep_detail_list(profile_public_id: str, cursor: str=None, date_end: str=None, page_size: int=None, date_start: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides sleep detail data."
    cursor: The pagination cursor value.
        date_end: Upper bound of date or timestamp.
        page_size: Number of results to return per page.
        date_start: Lower bound of date or timestamp.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/sleep_detail/"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if date_end:
        querystring['date_end'] = date_end
    if page_size:
        querystring['page_size'] = page_size
    if date_start:
        querystring['date_start'] = date_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profile_sleep_summary_list(profile_public_id: str, cursor: str=None, date_end: str=None, date_start: str=None, page_size: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides summarized sleep data."
    cursor: The pagination cursor value.
        date_end: Upper bound of date or timestamp.
        date_start: Lower bound of date or timestamp.
        page_size: Number of results to return per page.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/profile/{profile_public_id}/sleep_summary/"
    querystring = {}
    if cursor:
        querystring['cursor'] = cursor
    if date_end:
        querystring['date_end'] = date_end
    if date_start:
        querystring['date_start'] = date_start
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def challenge_members_read(public_id: str, limit: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a paginated list of members enrolled to a Challenge"
    limit: Number of results to return per page.
        offset: The initial index from which to return the results.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/challenge/{public_id}/members/"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def challenge_team_read(team_public_id: str, challenge_public_id: str, date_range: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List team members' contribution to the challenge.
		This endpoint returns a list of objects.
		"
    date_range: Index of the time period use
        
    """
    url = f"https://wefitter2.p.rapidapi.com/challenge/{challenge_public_id}/team/{team_public_id}/"
    querystring = {}
    if date_range:
        querystring['date_range'] = date_range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def challenge_leaderboard_read(public_id: str, date_range: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Displays the leaderboard for a challenge.
		This endpoint returns a list of objects.
		"
    date_range: Index of the time period use
        
    """
    url = f"https://wefitter2.p.rapidapi.com/challenge/{public_id}/leaderboard/"
    querystring = {}
    if date_range:
        querystring['date_range'] = date_range
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def challenge_leaderboard_history(public_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the scores of the authenticated profile for all periods since the profile joined the challenge.
		This endpoint returns a list of objects.
		"
    
    """
    url = f"https://wefitter2.p.rapidapi.com/challenge/{public_id}/leaderboard_history/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def challenge_team_list(challenge_public_id: str, limit: int=None, date_range: int=None, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the standings of different teams in challenges (don't have to be team challenges).
		This endpoint returns a list of objects.
		"
    limit: Number of results to return per page.
        date_range: Index of the time period use
        offset: The initial index from which to return the results.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/challenge/{challenge_public_id}/team/"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if date_range:
        querystring['date_range'] = date_range
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def challenge_read(public_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a single challenge"
    
    """
    url = f"https://wefitter2.p.rapidapi.com/challenge/{public_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def challenge_list(recent: int=None, offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all available challenges"
    recent: Show challenges that have not ended in the last N weeks
        offset: The initial index from which to return the results.
        limit: Number of results to return per page.
        
    """
    url = f"https://wefitter2.p.rapidapi.com/challenge/"
    querystring = {}
    if recent:
        querystring['recent'] = recent
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def validate_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all supported top-level schemas."
    
    """
    url = f"https://wefitter2.p.rapidapi.com/validate/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wefitter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

