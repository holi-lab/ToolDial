import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ad4_validatetoken(user_authorization: str, x_api_key: str, app_user_agent: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/app/users/validate-token"
    querystring = {'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e46_views_id_assessments_available_to_link(x_api_key: str, user_authorization: str, app_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/keys/626aef9fd6316f008f415bb4/pods/available-to-link"
    querystring = {'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e45_assessments_archived(x_api_key: str, app_authorization: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/block-pods/archived"
    querystring = {'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e44_assessments_archived(user_authorization: str, app_authorization: str, app_user_agent: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/pods/archived"
    querystring = {'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e19_blocks_id_pods_id(x_api_key: str, app_authorization: str, app_user_agent: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/blocks/626d9304d50156008fca40b0/pods/626d9331d501560090ca40b3"
    querystring = {'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def t6_courses_course_types_id(app_authorization: str, user_authorization: str, x_api_key: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/block-types/62670cf540239b0091e6b9ba/blocks"
    querystring = {'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def l3_badges_unread_status(x_api_key: str, app_user_agent: str, app_authorization: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/notifications/unread/count"
    querystring = {'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def c9_projectbytype(app_user_agent: str, x_api_key: str, user_authorization: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/keys/filtered/by-type"
    querystring = {'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def j1_views_id_notes(app_user_agent: str, app_authorization: str, x_api_key: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/keys/626aef9fd6316f008f415bb4/notes"
    querystring = {'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def c5_archivedprojects(app_user_agent: str, user_authorization: str, app_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/keys/archived"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def k8_conversations_unread_status(user_authorization: str, app_user_agent: str, x_api_key: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/dashboard/conversations/unread-status"
    querystring = {'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def j5_assessments_id_notes(app_user_agent: str, user_authorization: str, app_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/pods/626d9331d501560090ca40b3/notes"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def l1_badges(x_api_key: str, app_authorization: str, app_user_agent: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/notifications"
    querystring = {'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def c3_projects(user_authorization: str, app_authorization: str, x_api_key: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/keys"
    querystring = {'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def t3_course_types(app_user_agent: str, user_authorization: str, x_api_key: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/block-types/with/counts"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def l2_badges_unread(app_authorization: str, x_api_key: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/notifications/unread"
    querystring = {'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ab1_fetchblockattachments(user_authorization: str, app_user_agent: str, x_api_key: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/app/resource/attributes"
    querystring = {'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def a2_status(app_user_agent: str, x_api_key: str, user_authorization: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/app/status"
    querystring = {'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def c1_projectbyid(app_user_agent: str, user_authorization: str, app_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/keys/626c51beb2cc27008e097b66"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def j7_assessments_id_notes(app_authorization: str, user_authorization: str, x_api_key: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/block-pods/626d9331d501560090ca40b3/notes"
    querystring = {'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def k1_dashboard_combined_responses(app_user_agent: str, app_authorization: str, user_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/dashboard/combined-responses"
    querystring = {'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def k7_badges_unread_status(x_api_key: str, app_authorization: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/dashboard/badges/unread-status"
    querystring = {'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def p2_scheduler_all_events(user_authorization: str, x_api_key: str, app_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/scheduler/all-events/by-start-date"
    querystring = {'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def j3_courses_id_notes(app_user_agent: str, x_api_key: str, app_authorization: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/blocks/626d9304d50156008fca40b0/notes"
    querystring = {'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aa3_userprofile(user_authorization: str, x_api_key: str, app_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/profiles"
    querystring = {'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def p3_scheduler_standalone_events(app_authorization: str, app_user_agent: str, x_api_key: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/scheduler/standalone-events"
    querystring = {'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def k4_recently_modified_views(app_user_agent: str, app_authorization: str, user_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/dashboard/recently-modified/keys"
    querystring = {'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def k2_dashboard_recently_modified(app_authorization: str, user_authorization: str, x_api_key: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/dashboard/recently-modified"
    querystring = {'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def k6_dashboard_due_shortly_blocks(user_authorization: str, app_authorization: str, x_api_key: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/dashboard/due-shortly/blocks"
    querystring = {'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def c6_views_assessments_id_linked_to(app_user_agent: str, app_authorization: str, x_api_key: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/pods/626709db40239b0093e6b95d/linked-to/keys"
    querystring = {'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aa1_userintroduction(app_authorization: str, x_api_key: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/profiles/introduction"
    querystring = {'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def z3_templates_pods(user_authorization: str, app_authorization: str, x_api_key: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/templates/pods"
    querystring = {'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v19_charts_views_id_courses_id_linked_resources(app_authorization: str, app_user_agent: str, user_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/keys/627637e25dbe06008ad2d215/blocks/6276384c5dbe06008bd2d20f/linked-resources"
    querystring = {'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def i2_courses_id_comments(x_api_key: str, app_user_agent: str, app_authorization: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/blocks/626d9304d50156008fca40b0/comments"
    querystring = {'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def d24_search_courses_id_shareable_users(app_user_agent: str, user_authorization: str, x_api_key: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/search/blocks/626c53ddb2cc27008e097b67/shareable/users"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def i8_assessments_id_comments(user_authorization: str, app_user_agent: str, app_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/block-pods/626dd849d501560090ca40b7/comments"
    querystring = {'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def d13_courses_id_acl(x_api_key: str, app_user_agent: str, app_authorization: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/blocks/626c53ddb2cc27008e097b67/acl"
    querystring = {'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def i1_comments(app_user_agent: str, user_authorization: str, app_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/comments"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v20_charts_views_id_courses_id_grading_systems_id_grades(app_authorization: str, x_api_key: str, app_user_agent: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/keys/627637e25dbe06008ad2d215/blocks/6276384c5dbe06008bd2d20f/scales/627637e35dbe06008ad2d23b/grades"
    querystring = {'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v10_charts_views_id_filters(user_authorization: str, app_user_agent: str, x_api_key: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/keys/627637e25dbe06008ad2d215/filters"
    querystring = {'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def d8_views_id_courses_id_teachers(x_api_key: str, app_authorization: str, app_user_agent: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/keys/626c51beb2cc27008e097b66/blocks/626c53ddb2cc27008e097b67/teachers"
    querystring = {'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def w1_search(user_authorization: str, app_authorization: str, app_user_agent: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/search"
    querystring = {'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def d11_courses_id(user_authorization: str, x_api_key: str, app_user_agent: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/blocks/626c53ddb2cc27008e097b67"
    querystring = {'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def s8_assessments_grading_systems_id(x_api_key: str, user_authorization: str, app_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/scales/62705b1ddc14f30092429418/pods"
    querystring = {'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def d9_courses_id_teachers(app_user_agent: str, x_api_key: str, user_authorization: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/courses/626c53ddb2cc27008e097b67/teachers"
    querystring = {'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_charts_dashboard_filters(user_authorization: str, app_user_agent: str, app_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/dashboard/keys/filters"
    querystring = {'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e1_views_id_assessments(app_authorization: str, x_api_key: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/keys/626aef9fd6316f008f415bb4/pods"
    querystring = {'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e15_assessments_id(app_user_agent: str, user_authorization: str, app_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/pods/626c4f2eb2cc270090097b6d"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e17_assessments_id(x_api_key: str, app_user_agent: str, user_authorization: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/block-pods/626d9331d501560090ca40b3"
    querystring = {'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def m1_favorites(app_user_agent: str, user_authorization: str, x_api_key: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/favorites"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def y10_assessments_id_tasks(app_user_agent: str, user_authorization: str, x_api_key: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/block-pods/6275e071bfba3700ae364957/tasks"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def e4_courses_id_assessments(app_user_agent: str, app_authorization: str, x_api_key: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/blocks/626c5e76b2cc27008e097b69/pods"
    querystring = {'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ad5_facebooksignin(app_user_agent: str, app_authorization: str, x_api_key: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/app/auth/facebook/callback"
    querystring = {'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def x10_assessments_id_student_attachments(app_user_agent: str, user_authorization: str, app_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/teacher-pods/627639855dbe06008bd2d242/student/attachments/as-student"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def x1_courses_id_students(x_api_key: str, user_authorization: str, app_user_agent: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/blocks/6276384c5dbe06008bd2d20f/students"
    querystring = {'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def o2_conversations(app_user_agent: str, x_api_key: str, app_authorization: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/conversations"
    querystring = {'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def o6_conversations_id(x_api_key: str, user_authorization: str, app_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/conversations/62704952dc14f30091429410"
    querystring = {'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def x12_assessments_id_student_comments(app_authorization: str, x_api_key: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/teacher-pods/627639855dbe06008bd2d242/student/comments"
    querystring = {'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def x8_assessments_id_grades(app_user_agent: str, x_api_key: str, user_authorization: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/teacher-pods/627639855dbe06008bd2d242/student-grades/as-teacher"
    querystring = {'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def x5_courses_id_students_grades(x_api_key: str, app_authorization: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/blocks/6276384c5dbe06008bd2d20f/students/grades"
    querystring = {'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def x7_courses_id_grades(app_user_agent: str, app_authorization: str, user_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/blocks/6276384c5dbe06008bd2d20f/student-grades/as-teacher"
    querystring = {'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def x6_assessments_id_students_grades(x_api_key: str, app_user_agent: str, app_authorization: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/pods/627639855dbe06008bd2d242/students/grades"
    querystring = {'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def y1_views_id_tasks(app_authorization: str, x_api_key: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/keys/6275d20abfba3700af3648f0/tasks"
    querystring = {'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def x11_assessments_id_comments(user_authorization: str, app_user_agent: str, app_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/teacher-pods/627639855dbe06008bd2d242/comments/as-teacher"
    querystring = {'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ad3_refreshtoken(user_authorization: str, app_authorization: str, app_user_agent: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/app/users/refresh-token"
    querystring = {'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def t1_course_types(x_api_key: str, app_authorization: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/block-types"
    querystring = {'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def o1_conversations_unread_status(app_user_agent: str, user_authorization: str, app_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/conversations/unread-status"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def y4_courses_id_tasks(app_user_agent: str, app_authorization: str, user_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/blocks/6275d2f8bfba3700af364919/tasks"
    querystring = {'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def x9_assessments_id_attachments(x_api_key: str, user_authorization: str, app_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/teacher-pods/627639855dbe06008bd2d242/attachments/as-teacher"
    querystring = {'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def s6_grading_systems(app_authorization: str, x_api_key: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/scales/with/counts"
    querystring = {'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def d1_blocksinproject(app_user_agent: str, x_api_key: str, user_authorization: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/keys/626c5d83b2cc27008e097b68/blocks"
    querystring = {'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def d3_podslinkedtoblocks(x_api_key: str, app_authorization: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/pods/626c5eb1b2cc270090097b70/linked-to/blocks"
    querystring = {'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v7_charts_dashboard_grading_systems(app_user_agent: str, x_api_key: str, user_authorization: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/dashboard/scales"
    querystring = {'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def d10_views_id_courses_available_to_link(x_api_key: str, user_authorization: str, app_user_agent: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/keys/626c51beb2cc27008e097b66/blocks/available-to-link"
    querystring = {'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def s1_grading_systems(x_api_key: str, app_authorization: str, app_user_agent: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/scales"
    querystring = {'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def s7_courses_grading_systems_id(app_authorization: str, x_api_key: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/scales/62705b1ddc14f30092429418/blocks"
    querystring = {'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def i12_assessments_id_student_comments(user_authorization: str, app_authorization: str, x_api_key: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/pods/627639855dbe06008bd2d242/submissions/comments/as-student"
    querystring = {'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def w2_search_users(app_authorization: str, x_api_key: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/search/users"
    querystring = {'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def s3_grading_systems_id(app_authorization: str, x_api_key: str, app_user_agent: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/scales/62705bccdc14f3009142941a"
    querystring = {'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def i5_assessments_id_comments(app_user_agent: str, app_authorization: str, user_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/pods/626dd849d501560090ca40b7/comments"
    querystring = {'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def g1_keys_id_checklists(x_api_key: str, app_user_agent: str, app_authorization: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/keys/6278c2a25a3b5d008ad1902c/checklists"
    querystring = {'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def g6_blocks_id_checklists(app_user_agent: str, user_authorization: str, app_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/blocks/6278c2b35a3b5d008bd1902d/checklists"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def g11_pods_id_checklists(app_user_agent: str, user_authorization: str, x_api_key: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/pods/627d4baea710be008d9c91e3/checklists"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def y7_assessments_id_tasks(app_authorization: str, user_authorization: str, app_user_agent: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/pods/6275de6bbfba3700b036492d/tasks"
    querystring = {'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def o4_conversations(user_authorization: str, x_api_key: str, app_user_agent: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/conversations/by-usernames"
    querystring = {'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def p1_scheduler_all_events(app_user_agent: str, x_api_key: str, user_authorization: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/scheduler/all-events"
    querystring = {'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v9_charts_views_id_courses_assessments(x_api_key: str, app_authorization: str, app_user_agent: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/keys/627637e25dbe06008ad2d215"
    querystring = {'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def a1_latestversion(app_user_agent: str, app_authorization: str, x_api_key: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/app/latest-version"
    querystring = {'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v8_charts_dashboard_task_status(app_authorization: str, x_api_key: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/dashboard/task-status"
    querystring = {'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v4_charts_dashboard_system_filters(x_api_key: str, user_authorization: str, app_user_agent: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/dashboard/system-keys/filters"
    querystring = {'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v13_charts_views_id_grading_systems(x_api_key: str, app_authorization: str, app_user_agent: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/keys/627637e25dbe06008ad2d215/scales"
    querystring = {'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aa2_studentprofile(user_authorization: str, x_api_key: str, app_user_agent: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/profiles/users/627638575dbe06008bd2d212"
    querystring = {'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def aa5_searchuserswithsearchtokenforusername(user_authorization: str, app_user_agent: str, x_api_key: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/search/users/this"
    querystring = {'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v11_charts_views_id_course_types(app_authorization: str, app_user_agent: str, x_api_key: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/keys/627637e25dbe06008ad2d215/block-types"
    querystring = {'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v5_charts_dashboard_course_types(app_authorization: str, user_authorization: str, app_user_agent: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/dashboard/block-types"
    querystring = {'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def k5_dashboard_due_shortly(x_api_key: str, app_user_agent: str, user_authorization: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/dashboard/due-shortly/resources"
    querystring = {'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v18_charts_views_id_course_teachers(x_api_key: str, app_user_agent: str, app_authorization: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/keys/627637e25dbe06008ad2d215/teachers"
    querystring = {'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def c7_blocksbyprojectid(x_api_key: str, app_authorization: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/blocks/6276384c5dbe06008bd2d20f/linked-to/keys"
    querystring = {'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v14_charts_views_id_linked_resources(app_user_agent: str, user_authorization: str, app_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/keys/627637e25dbe06008ad2d215/linked-resources"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def u1_assessment_types(x_api_key: str, user_authorization: str, app_user_agent: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/pod-types"
    querystring = {'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def z2_templates_blocks(app_user_agent: str, app_authorization: str, user_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/templates/blocks"
    querystring = {'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def d29_courses_archived(app_user_agent: str, user_authorization: str, x_api_key: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/blocks/archived"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v15_charts_views_id_grading_systems_id_grades(user_authorization: str, app_user_agent: str, x_api_key: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/keys/627637e25dbe06008ad2d215/scales/627637e35dbe06008ad2d23b/scales"
    querystring = {'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v12_charts_views_id_assessment_types(app_authorization: str, x_api_key: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/keys/627637e25dbe06008ad2d215/pod-types"
    querystring = {'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def k3_unread_count(x_api_key: str, app_authorization: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/dashboard/unread-count"
    querystring = {'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def u6_assessments_assessment_types_id(app_authorization: str, user_authorization: str, app_user_agent: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/pod-types/626701a440239b0091e6b9b1/pods"
    querystring = {'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v16_charts_views_id_task_status(user_authorization: str, app_authorization: str, x_api_key: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/keys/627637e25dbe06008ad2d215/task-status"
    querystring = {'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def z1_templates_keys(x_api_key: str, app_authorization: str, user_authorization: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/templates/keys"
    querystring = {'x-api-key': x_api_key, 'App-Authorization': app_authorization, 'User-Authorization': user_authorization, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_charts_dashboard_views_courses_assessments(app_authorization: str, x_api_key: str, app_user_agent: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/dashboard/keys"
    querystring = {'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def u3_assessment_types(user_authorization: str, app_authorization: str, x_api_key: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/pod-types/with/counts"
    querystring = {'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def i11_assessments_id_comments(app_user_agent: str, x_api_key: str, user_authorization: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/pods/627639855dbe06008bd2d242/submissions/comments/as-teacher"
    querystring = {'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_charts_dashboard_system_views_courses_assessments(app_user_agent: str, user_authorization: str, app_authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/dashboard/system-keys"
    querystring = {'App-user-agent': app_user_agent, 'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v21_charts_views_id_courses_id_task_status(app_authorization: str, app_user_agent: str, x_api_key: str, user_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/keys/627637e25dbe06008ad2d215/blocks/6276384c5dbe06008bd2d20f/task-status"
    querystring = {'App-Authorization': app_authorization, 'App-user-agent': app_user_agent, 'x-api-key': x_api_key, 'User-Authorization': user_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v17_charts_views_id_course_students(user_authorization: str, app_authorization: str, x_api_key: str, app_user_agent: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/keys/627637e25dbe06008ad2d215/students"
    querystring = {'User-Authorization': user_authorization, 'App-Authorization': app_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v6_charts_dashboard_assessment_types(user_authorization: str, x_api_key: str, app_user_agent: str, app_authorization: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://pitch-testing.p.rapidapi.com/charts/dashboard/pod-types"
    querystring = {'User-Authorization': user_authorization, 'x-api-key': x_api_key, 'App-user-agent': app_user_agent, 'App-Authorization': app_authorization, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pitch-testing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

