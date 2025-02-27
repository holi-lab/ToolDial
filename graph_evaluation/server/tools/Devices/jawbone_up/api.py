import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_generic_event_for_the_user(date: str=None, page_token: str=None, start_time: str=None, end_time: str=None, updated_after: str=None, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of generic events of the current user. This list can be paginated by date or timestamp."
    date: Date, formatted as YYYYMMDD. If omitted, returns the information for today.
        page_token: Timestamp used to paginate the list of events. The Developer must use the "next" link provided in the "links" section.
        start_time: To be used along with end_time. Epoch timestamp that denotes the start of the time range queried for events.
        end_time: To be used with start_time. Epoch timestamp that denotes the end of the time range queried for events.
        updated_after: Epoch timestamp to list events that are updated later than the timestamp. To be used with start_time to list events that were completed after said start_time.
        limit: Maximum number of results to return
        
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/users/@me/generic_events"
    querystring = {}
    if date:
        querystring['date'] = date
    if page_token:
        querystring['page_token'] = page_token
    if start_time:
        querystring['start_time'] = start_time
    if end_time:
        querystring['end_time'] = end_time
    if updated_after:
        querystring['updated_after'] = updated_after
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_single_body_composition_metric_event(event_xid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/body_events/{event_xid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_single_cardiac_metric_event(event_xid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/cardiac_events/{event_xid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_body_metrics_record_events(date: str, page_token: str, start_time: str, end_time: str, updated_after: str, limit: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    date: Date, formatted as YYYYMMDD. If omitted, returns the information for today..
        page_token: Timestamp used to paginate the list of sleeps. The Developer must use the "next" link provided in the "links" section.
        start_time: To be used along with end_time. Epoch timestamp that denotes the start of the time range queried for events.
        end_time: To be used with start_time. Epoch timestamp that denotes the end of the time range queried for events.
        updated_after: Epoch timestamp to list events that have been updated later than the timestamp.
        limit: Maximum number of results to return
        
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/users/@me/body_events"
    querystring = {'date': date, 'page_token': page_token, 'start_time': start_time, 'end_time': end_time, 'updated_after': updated_after, 'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_single_mood_event(xid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/mood/{xid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_cardiac_metrics_events_for_a_user(date: str=None, page_token: str=None, start_time: str=None, end_time: str=None, updated_after: str=None, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    date: Date, formatted as YYYYMMDD. If omitted, returns the information for today..
        page_token: Timestamp used to paginate the list of events. The Developer must use the "next" link provided in the "links" section.
        start_time: To be used along with end_time. Epoch timestamp that denotes the start of the time range queried for events.
        end_time: To be used with start_time. Epoch timestamp that denotes the end of the time range queried for events.
        updated_after: Epoch timestamp to list events that have been updated later than the timestamp.
        limit: Maximum number of results to return
        
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/users/@me/cardiac_events"
    querystring = {}
    if date:
        querystring['date'] = date
    if page_token:
        querystring['page_token'] = page_token
    if start_time:
        querystring['start_time'] = start_time
    if end_time:
        querystring['end_time'] = end_time
    if updated_after:
        querystring['updated_after'] = updated_after
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_information_about_specific_meal(xid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns detailed information about a specific meal"
    
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/meals/{xid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_detailed_information_about_the_user(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the basic information of the user"
    
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/users/@me"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sleep_phases(xid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A sleep period can be sub-divided into phases: awake, light and deep. This endpoint returns a time-series of the sleep phase during the period. Each entry is a tuple that contains a timestamp and the sleep phase (1=awake, 2=light, 3=deep). The timestamps are distributed based on when the type of sleep changes."
    
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/sleeps/{xid}/snapshot"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_friends(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of unique identifiers (XIDs) of the user's friends."
    
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/users/@me/friends"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_move_intensity(move_xid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a time-series of the move intensity during the day. Each entry is a tuple that contains a timestamp, and the average intensity during that period. Currently, it’s an arbitrary equidistributed list of timestamps."
    
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/moves/{move_xid}/snapshot"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_sleep_period_information(xid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns detailed information about the given sleep period denoted by xid."
    
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/sleeps/{xid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_information_about_a_specific_workout(xid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns detailed information about the given workout."
    
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/workouts/{xid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_information_about_a_specific_move(move_xid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the detailed information of the given move."
    
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/moves/{move_xid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_user_s_move_graphs(move_xid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the image of the given move."
    
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/moves/{move_xid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_user_s_moves_list(date: str=None, page_token: str=None, start_time: str=None, end_time: str=None, updated_after: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of moves of the current user. This list can be paginated by date or timestamp."
    date: Date, formatted as YYYYMMDD. If omitted, returns the information for today.
        page_token: Timestamp used to paginate the list of moves. The Developer must use the “next” link provided in the “links” section.
        start_time: To be used along with end_time. Epoch timestamp that denotes the start of the time range queried for move events.
        end_time: To be used with start_time. Epoch timestamp that denotes the end of the time range queried for move events.
        updated_after: Epoch timestamp to list move events that are updated later than the timestamp. To be used with start_time to list events that were completed after said start_time.
        
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/users/@me/moves"
    querystring = {}
    if date:
        querystring['date'] = date
    if page_token:
        querystring['page_token'] = page_token
    if start_time:
        querystring['start_time'] = start_time
    if end_time:
        querystring['end_time'] = end_time
    if updated_after:
        querystring['updated_after'] = updated_after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_user_s_trends(end_date: str=None, range_duration: str=None, range: str=None, bucket_size: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the user's trends over a period of time (e.g. 5 weeks), using the given granularity (e.g. by day)."
    end_date: Date, formatted as YYYYMMDD. If omitted, returns until today.
        range_duration: Used with range to determine how long to go back in time.
        range: Used with range_duration to determine how long to go back in time. Possible values are: d (for “days”) and w (for “weeks”).
        bucket_size: Determines the granularity to use when aggregating the values. Possible values are: d (for “days”), w (for “weeks”), m (for “months”), y (for “years”).
        
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/users/@me/trends"
    querystring = {}
    if end_date:
        querystring['end_date'] = end_date
    if range_duration:
        querystring['range_duration'] = range_duration
    if range:
        querystring['range'] = range
    if bucket_size:
        querystring['bucket_size'] = bucket_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_user_s_mood(date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the user's mood for the given day. Defaults to get today's mood."
    date: Date, formatted as YYYYMMDD. If omitted, returns today's mood.
        
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/users/@me/mood"
    querystring = {}
    if date:
        querystring['date'] = date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_user_s_workout_graphs(xid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/workouts/{xid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_s_meal_list(date: str=None, page_token: str=None, start_time: str=None, end_time: str=None, updated_after: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of meals of the current user. This list can be paginated by date or timestamp."
    date: Date, formatted as YYYYMMDD. If omitted, returns the information for today.
        page_token: Timestamp used to paginate the list of meals. The Developer must use the "next" link provided in the "links" section.
        start_time: To be used along with end_time. Epoch timestamp that denotes the start of the time range queried for events.
        end_time: To be used with start_time. Epoch timestamp that denotes the end of the time range queried for events.
        updated_after: Epoch timestamp to list events that are updated later than the timestamp. To be used with start_time to list events that were completed after said start_time.
        
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/users/@me/meals"
    querystring = {}
    if date:
        querystring['date'] = date
    if page_token:
        querystring['page_token'] = page_token
    if start_time:
        querystring['start_time'] = start_time
    if end_time:
        querystring['end_time'] = end_time
    if updated_after:
        querystring['updated_after'] = updated_after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_user_s_workout_list(date: str=None, page_token: str=None, start_time: str=None, end_time: str=None, updated_after: str=None, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of workouts of the current user. This list can be paginated by date or timestamp."
    date: Date, formatted as YYYYMMDD. If omitted, returns the information for today
        page_token: Timestamp used to paginate the list of workouts. The Developer must use the "next" link provided in the "links" section.
        start_time: To be used along with end_time. Epoch timestamp that denotes the start of the time range queried for events.
        end_time: To be used with start_time. Epoch timestamp that denotes the end of the time range queried for events.
        updated_after: Epoch timestamp to list events that are updated later than the timestamp. To be used with start_time to list events that were completed after said start_time.
        limit: Maximum number of results to return
        
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/users/@me/workouts"
    querystring = {}
    if date:
        querystring['date'] = date
    if page_token:
        querystring['page_token'] = page_token
    if start_time:
        querystring['start_time'] = start_time
    if end_time:
        querystring['end_time'] = end_time
    if updated_after:
        querystring['updated_after'] = updated_after
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_s_sleep_graphs(xid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the image of the given sleep."
    
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/sleeps/{xid}/image"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_workout_intensity(xid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a time-series of the workout intensity during the day. Each entry is a tuple that contains a timestamp, and the average intensity during that period. Currently, it's an arbitrary equidistributed list of timestamps."
    
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/workouts/{xid}/snapshot"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_s_sleep(date: str=None, page_token: str=None, start_time: str=None, end_date: str=None, updated_after: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of sleeps of the current user. This list can be paginated by date or timestamp."
    date: Date, formatted as YYYYMMDD. If omitted, returns the information for today.
        page_token: Timestamp used to paginate the list of sleeps. The Developer must use the "next" link provided in the "links" section.
        start_time: To be used along with end_time. Epoch timestamp that denotes the start of the time range queried for events.
        end_date: To be used with start_time. Epoch timestamp that denotes the end of the time range queried for events.
        updated_after: Epoch timestamp to list events that are updated later than the timestamp. To be used with start_time to list events that were completed after said start_time.
        
    """
    url = f"https://community-jawbone-up.p.rapidapi.com/users/@me/sleeps"
    querystring = {}
    if date:
        querystring['date'] = date
    if page_token:
        querystring['page_token'] = page_token
    if start_time:
        querystring['start_time'] = start_time
    if end_date:
        querystring['end_date'] = end_date
    if updated_after:
        querystring['updated_after'] = updated_after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "community-jawbone-up.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

