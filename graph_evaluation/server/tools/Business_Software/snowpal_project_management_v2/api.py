import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_notifications(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all notifications."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/notifications"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_user_by_token(token: str='rapid', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches user by token."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/search/users"
    querystring = {}
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_events_in_given_window(enddate: str='2023-03-17T15:38:07.394Z', startdate: str='2023-01-17T15:38:07.394Z', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all events in given window."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/scheduler/all-events"
    querystring = {}
    if enddate:
        querystring['endDate'] = enddate
    if startdate:
        querystring['startDate'] = startdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_events_for_given_day(startdate: str='2023-02-17T15:38:07.394Z', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all events for given day."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/scheduler/all-events/by-start-date"
    querystring = {}
    if startdate:
        querystring['startDate'] = startdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_pods(is_id: str, batchindex: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pods. If `batchIndex` is not specified, all pods will be returned in one go."
    batchindex: Specify index of result set (aka, page number).
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/keys/{is_id}/pods"
    querystring = {}
    if batchindex:
        querystring['batchIndex'] = batchindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_scales(includecounts: str=None, excludeempty: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all scales."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/scales"
    querystring = {}
    if includecounts:
        querystring['includeCounts'] = includecounts
    if excludeempty:
        querystring['excludeEmpty'] = excludeempty
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block(is_id: str, keyid: str='63ef8da92a51fc00376046e0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block for given ID."
    keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/blocks/{is_id}"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pod_grades_for_all_students(is_id: str, keyid: str='63ceb29edb035900138d975d', blockid: str='93bwn23fje782486247d248h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all students' teacher block grades for a given pod."
    keyid: Specify Key ID.
        blockid: Specify Block ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/classroom-pods/{is_id}/students/grades"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_grades_for_all_students(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all students' teacher block grades for a given block."
    keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/classroom-blocks/{is_id}/students/grades"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_student_profile(is_id: str, keyid: str='63ceb29edb035900138d975d', blockid: str='98nbc29nbr835753138d370c', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns student profile. This is specific to a Teacher Key."
    keyid: Specify Key ID.

        blockid: Specify Block ID.

        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/classroom/students/{is_id}/profile"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_students_in_a_block(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all students in a block."
    keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/classroom-blocks/{is_id}/students"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_attachment_submissions_as_student(is_id: str, blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns attachment submissions (as student)"
    blockid: Specify Block ID.
        keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/classroom-pods/{is_id}/submissions/attachments/as-student"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pods_using_pod_type(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns pods using given pod type."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/pod-types/{is_id}/pods"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_standalone_events(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns standalone events."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/scheduler/standalone-events"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_scale(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a scale."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/scales/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pod_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns pod types. Set `includeCounts` to _true_ if you want a count of associated pods."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/pod-types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_relations_matching_search_token(currentblockid: str='93bwn23fje782486247d248h', currentpodid: str='63ceb29edb035900138d975d', blockid: str='93bwn23fje782486247d248h', token: str='token', currentkeyid: str='63ceb29edb035900138d975d', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns relations matching search token."
    currentblockid: Specify current block ID.

        currentpodid: Specify current pod ID.

        blockid: Specify Block ID (only needed for Block Pods)
        token: Specify search token.

        currentkeyid: Specify current key ID.

        keyid: Specify Key ID (only needed for Block Pods)
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/search/relations"
    querystring = {}
    if currentblockid:
        querystring['currentBlockId'] = currentblockid
    if currentpodid:
        querystring['currentPodId'] = currentpodid
    if blockid:
        querystring['blockId'] = blockid
    if token:
        querystring['token'] = token
    if currentkeyid:
        querystring['currentKeyId'] = currentkeyid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pod_relations(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns relations for given pod ID."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/pods/{is_id}/relations"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_relations(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns relations for given block ID."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/blocks/{is_id}/relations"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_unread_notifications(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns unread notifications."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/notifications/unread"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_pod_relations(is_id: str, keyid: str='63ceb29edb035900138d975d', blockid: str='93bwn23fje782486247d248h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns relations for given block pod ID."
    keyid: Specify Key ID
        blockid: Specify Block ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/block-pods/{is_id}/relations"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_relations(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns relations for given key ID."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/keys/{is_id}/relations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_pod_tasks(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod tasks."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/pods/{is_id}/tasks"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_pod_notes(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod notes. A note is private to the user."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/pods/{is_id}/notes"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_pod(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod."
    keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/pods/{is_id}"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_pods_available_to_be_linked(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pods available to be linked to this key. A pod can only be linked once to a key but it can be linked to any number of keys, and/or blocks."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/keys/{is_id}/pods/available-to-link"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_project_list(list_id: str, is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns project list for given ID."
    keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/blocks/{is_id}/project-block-lists/{list_id}"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_project_lists(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns project lists."
    keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/blocks/{is_id}/project-block-lists"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_archived_pods(keyid: str='67mnw82huw218472984b398h', batchindex: str='batchIndex', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns archived pods."
    keyid: Specify Key ID.
        batchindex: Specify index of result set (aka, page number).
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/pods/archived"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if batchindex:
        querystring['batchIndex'] = batchindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_checklists(is_id: str, keyid: str='63ef8da92a51fc00376046e0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block checklists."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/blocks/{is_id}/checklists"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_pod_comments(is_id: str, keyid: str='63ef8da92a51fc00376046e0', blockid: str='63ef8e4c2a51fc0035604647', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod comments."
    keyid: Specify Key ID.
        blockid: Specify Block ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/block-pods/{is_id}/comments"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_s_profile(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns user's profile."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/profiles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_s_introduction_profile(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns user's introduction profile. This returns attributes that are pertinent to user's sign up, and typically used following a recent sign up."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/profiles/introduction"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_app_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns app status."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/app/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_version(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns latest version."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/app/latest-version"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_by_uuid(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns user by UUID."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/users/uuid/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_templates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block templates."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/templates/blocks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pod_templates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns pod templates."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/templates/pods"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_and_pods_grades_for_a_student_as_teacher(is_id: str, keyid: str='63ceb29edb035900138d975d', studentuserid: str='84ytd25kdb035909838d992k', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block and pods' grades for a student. This request should be made by a teacher."
    keyid: Specify Key ID.
        studentuserid: Specify student ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/classroom-blocks/{is_id}/student-grades/as-teacher"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if studentuserid:
        querystring['studentUserId'] = studentuserid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_student_attachment_submissions_as_teacher(is_id: str, studentid: str='84ytd25kdb035909838d992k', blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns student attachment submissions (as teacher)."
    studentid: Specify student ID.
        blockid: Specify Block ID.
        keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/classroom-pods/{is_id}/submissions/attachments/as-teacher"
    querystring = {}
    if studentid:
        querystring['studentId'] = studentid
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_student_comment_submissions_as_teacher(is_id: str, keyid: str='63ceb29edb035900138d975d', studentid: str='84ytd25kdb035909838d992k', blockid: str='93bwn23fje782486247d248h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns student comment submissions (as teacher)."
    keyid: Specify Key ID.
        studentid: Specify student ID.
        blockid: Specify Block ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/classroom-pods/{is_id}/submissions/comments/as-teacher"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if studentid:
        querystring['studentId'] = studentid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_pod_and_block_scale_values(scale_id: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod and block scale values using given scale."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/keys/{is_id}/scales/{scale_id}/scale-values"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_blocks_and_pods_associated_with_key(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns blocks and pods associated with given key."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/keys/{is_id}/blocks-pods"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_pod_checklists(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod checklists."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/pods/{is_id}/checklists"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_filtered_user_keys_blocks_and_pods_for_given_key(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns `created by me`, `shared with me`, and `shared with others` filter results for given key."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/keys/{is_id}/filters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_linked_resources(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns blocks & key pods linked to keys, and key pods linked to blocks."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/keys/{is_id}/linked-resources"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_pod_attachments(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod attachments."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/pods/{is_id}/attachments"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_types_and_blocks_based_on_them_in_key(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all block types along with blocks in given key that use those block types."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/keys/{is_id}/block-types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pods_based_on_pod_types_in_key(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all pod types along with pods in given key that use those pod types."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/keys/{is_id}/pod-types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key for given ID."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/keys/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pods_and_tasks_due_shortly(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns pods and tasks due shortly."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/dashboard/due-shortly/pods-and-tasks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_filtered_user_keys_blocks_and_pods(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns `created by me`, `shared with me`, and `shared with others` filter results for all keys except system keys."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/dashboard/keys/filters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pods_based_on_pod_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all pod types along with pods that use those pod types."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/dashboard/pod-types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_blocks_based_on_block_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all block types along with blocks that use those block types."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/dashboard/block-types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_system_keys_blocks_and_pods(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns system keys, blocks and pods."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/dashboard/system-keys"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tasks_by_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a breakdown of tasks (by status) across all keys."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/dashboard/task-status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_keys_blocks_and_pods(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns user keys, blocks and pods."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/dashboard/keys-blocks-pods"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_pod_collaborators(is_id: str, keyid: str='63ef8da92a51fc00376046e0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod collaborators. 
		
		Alternate endpoint - `/pods/:id/collaborators`."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/pods/{is_id}/acl"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_pod_checklists(is_id: str, blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod checklists. A block pod can have any number of checklists."
    blockid: Specify Block ID
        keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/block-pods/{is_id}/checklists"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_pods(is_id: str, batchindex: str='batchIndex', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If you want to fetch block pods in batches (_of 16_), set `batchIndex` to _true_. Else, all block pods will be returned in one go."
    batchindex: Specify index of result set (aka, page number).
        keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/blocks/{is_id}/pods"
    querystring = {}
    if batchindex:
        querystring['batchIndex'] = batchindex
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_tasks(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block tasks."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/blocks/{is_id}/tasks"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_notes(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block note. A note is private to you and therefore, will not be available to your collaborators even if the block is shared."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/blocks/{is_id}/notes"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_comments(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block comments. If the block is shared with other collaborators, the comment will be available to them as well."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/blocks/{is_id}/comments"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_linked_block_pods(block_id: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get block pods linked to given block or key."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/keys/{is_id}/blocks/{block_id}/linked-resources"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_grades_for_all_students_for_charts(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns grades of all students for given block."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/blocks/{is_id}/students/all/grades"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_attachments(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns attachments associated with the given block."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/blocks/{is_id}/attachments"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_blocks_available_to_be_linked_to_this_key(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A block can be linked to any number of keys but it can only be linked to a key once. Use this endpoint to get a list of all blocks that are available to be linked to a given key."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/keys/{is_id}/blocks/available-to-link"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_blocks_linked_to_pod(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A pod can be linked to one or more blocks. This endpoint returns all blocks a given pod is linked to."
    keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/pods/{is_id}/linked-to/blocks"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_blocks_in_a_key(is_id: str, aclwriteorhigher: bool=None, filter: str='filter', batchindex: str='batchIndex', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all blocks in a key."
    aclwriteorhigher: - True, if ACL is write or higher.
- False, otherwise.

        filter: Specify filter.
        batchindex: Specify index of result set (aka, page number).
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/keys/{is_id}/blocks"
    querystring = {}
    if aclwriteorhigher:
        querystring['aclWriteOrHigher'] = aclwriteorhigher
    if filter:
        querystring['filter'] = filter
    if batchindex:
        querystring['batchIndex'] = batchindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_archived_blocks(batchindex: str='batchIndex', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all archived blocks in given key."
    batchindex: Specify index of result set (aka, page number).
        keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/blocks/archived"
    querystring = {}
    if batchindex:
        querystring['batchIndex'] = batchindex
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_displayable_attributes_of_key_block_and_pod(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns displayable attributes of Key, Block and Pod."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/app/resource/attributes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_task(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key tasks."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/keys/{is_id}/tasks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_favorites(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all favorites - keys, blocks, key pods and block pods."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/favorites"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_recently_modified_keys(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns recently modified keys."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/dashboard/recently-modified/keys"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_unread_notifications(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns unread notifications."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/dashboard/notifications/unread-status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_unread_count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns unread count for -
		  - Notifications 
		  - Conversations
		  - Blocks Due
		  - Pods Due
		  - Tasks Due"
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/dashboard/unread-count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_dashboard_details(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the following dashboard details -
		  - Recently modified blocks
		  - Recently modified pods
		  - Pods due
		  - Tasks due"
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/dashboard/combined-responses"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_blocks_and_pods_based_on_scales(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all scales along with both blocks and pods that use those scales."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/dashboard/scales"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_filtered_system_keys_blocks_and_pods(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns `created by me`, `shared with me`, and `shared with others` filter results for all system keys."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/dashboard/system-keys/filters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_conversation(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns existing conversation for given ID."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/conversations/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_conversation_for_given_usernames(usernames: str='user1,user2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns conversation for given usernames (if there happens to be one)."
    usernames: Comma-separated list of recipients' user names.

        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/conversations/by-usernames"
    querystring = {}
    if usernames:
        querystring['userNames'] = usernames
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user_conversations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns user conversations."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/conversations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_unread_conversations_count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns count of unread conversations (aka, conversations that have at least one message that user has not yet read)."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/conversations/unread-status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_users_this_key_pod_can_be_shared_with(is_id: str, token: str='token', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A key pod can be shared with a user only once. So, when you search for a user, this will return all users that match your search query that do not already have access to the key pod."
    token: Specify search token.

        keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/search/pods/{is_id}/shareable/users"
    querystring = {}
    if token:
        querystring['token'] = token
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_blocks_using_block_type(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns blocks using given block type."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/block-types/{is_id}/blocks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_pod_tasks(is_id: str, keyid: str='63ceb29edb035900138d975d', blockid: str='93bwn23fje782486247d248h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod tasks."
    keyid: Specify Key ID
        blockid: Specify Block ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/block-pods/{is_id}/tasks"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_users_this_block_pod_can_be_shared_with(is_id: str, blockid: str='93bwn23fje782486247d248h', token: str='token', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A block pod can be shared with a user only once. So, when you search for a user, this will return all users that match your search query that do not already have access to the block pod."
    blockid: Specify Block ID
        token: Specify search token.

        keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/search/block-pods/{is_id}/shareable/users"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if token:
        querystring['token'] = token
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_pod_collaborators(is_id: str, blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod collaborators. 
		
		Alternate endpoint - `/blocks/:id/pods/:id/acl`."
    blockid: Specify Block ID
        keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/block-pods/{is_id}/acl"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_collaborators(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all users who have access to a given block. Depending on where the block is located, there will be differences in access levels.
		
		- A block in a Teacher Key can be granted _admin_ or _read_ access. By granting admin access, you make that user a _Teacher_. If you granted _read_ access, the user will become a *Student*.
		- A block in a Project Key can be granted _admin_, _write_, or _read_ access. If you want the collaborator to be able to share the block in turn, grant them _admin_ access. Else, you can grant them _write_, or _read_."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/blocks/{is_id}/acl"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_recent_comments(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns recent comments."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/comments"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block types. Set `includeCounts` to _true_ if you want a count of associated blocks."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/block-types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_pod_notes(is_id: str, blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod notes. A note is private to a user, and is therefore not available to other collaborators."
    blockid: Specify Block ID
        keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/block-pods/{is_id}/notes"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_pod_grades_for_all_students(is_id: str, blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod grades for all students."
    blockid: Specify Block ID
        keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/classroom-pods/{is_id}/students/grades"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_pod_tasks_for_charts(is_id: str, keyid: str='63ceb29edb035900138d975d', blockid: str='93bwn23fje782486247d248h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod tasks."
    keyid: Specify Key ID
        blockid: Specify Block ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/block-pods/{is_id}/tasks"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_pod(is_id: str, keyid: str='63ceb29edb035900138d975d', blockid: str='93bwn23fje782486247d248h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod. Alternate endpoint - `blocks/:id/pods/:id`."
    keyid: Specify Key ID.
        blockid: Specify Block ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/block-pods/{is_id}"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_archived_block_pods(batchindex: str='batchIndex', blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns archived block pods for given block."
    batchindex: Specify index of result set (aka, page number).
        blockid: Specify Block ID.
        keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/block-pods/archived"
    querystring = {}
    if batchindex:
        querystring['batchIndex'] = batchindex
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_blocks_due_shortly(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns blocks due shortly."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/dashboard/due-shortly/blocks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_recently_modified_blocks_and_pods(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns recently modified blocks and pods."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/dashboard/recently-modified"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_keys_a_block_is_linked_to(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns keys a given block is linked to."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/blocks/{is_id}/linked-to/keys"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_archived_keys(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns archived keys. Keys can be unarchived."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/keys/archived"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_keys_filtered_by_type(keytype: str='keyType', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns keys filtered by given type, as defined in `keyType`."
    keytype: Specify type of key.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/keys/filtered/by-type"
    querystring = {}
    if keytype:
        querystring['keyType'] = keytype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_keys_a_pod_is_linked_to(is_id: str, blockid: str='93bwn23fje782486247d248h', keyid: str='83yuz28ksn823022876m107v', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns keys a given pod is linked to."
    blockid: Specify Block ID.
        keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/pods/{is_id}/linked-to/keys"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_keys(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all keys. A key is the starting point, and is private to a user. Think of a key as a project, or a board."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/keys"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_key_block_or_pod_by_token(token: str='rapd', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches key, block or pod by token."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/search"
    querystring = {}
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_users_this_block_can_be_shared_with(is_id: str, keyid: str='63ceb29edb035900138d975d', token: str='token', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A block can be shared with a user only once. So, when you search for a user, this will return all users that match your search query that do not already have access to the block."
    keyid: Specify Key ID
        token: Specify search token.

        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/search/blocks/{is_id}/shareable/users"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_unread_notification_count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns unread notification count."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/notifications/unread/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_notes(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key notes."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/keys/{is_id}/notes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_task_status(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a breakdown of tasks in given key."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/keys/{is_id}/task-status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_comment_submissions_as_student(is_id: str, keyid: str='63ceb29edb035900138d975d', blockid: str='93bwn23fje782486247d248h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns comment submissions (as student)."
    keyid: Specify Key ID.
        blockid: Specify Block ID.
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/classroom-pods/{is_id}/submissions/comments/as-student"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_pods_using_scale(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns pods using given scale."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/scales/{is_id}/pods"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_users(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all users."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/users"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_checklists(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key checklists."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/keys/{is_id}/checklists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_pod_comments(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod comments. A comment is available to all collaborators."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/pods/{is_id}/comments"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_key_templates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key templates."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/templates/keys"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_blocks_using_scale(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns blocks using given scale."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/scales/{is_id}/blocks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_unread_conversations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns unread conversations."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/dashboard/conversations/unread-status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_scale_values_for_scale(scale_id: str, block_id: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block scale values for associated scale."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/keys/{is_id}/blocks/{block_id}/scales/{scale_id}/grades"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_task_status_for_block(block_id: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns task status for given block."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/keys/{is_id}/blocks/{block_id}/task-status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_scales_along_with_blocks_and_pods_based_on_them(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all scales along with both blocks and pods in given key that use those scales."
    
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/charts/keys/{is_id}/scales"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_block_pod_attachments(is_id: str, keyid: str='63ceb29edb035900138d975d', blockid: str='93bwn23fje782486247d248h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns pod attachments."
    keyid: Specify Key ID
        blockid: Specify Block ID
        
    """
    url = f"https://snowpal-project-management1.p.rapidapi.com/block-pods/{is_id}/attachments"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

