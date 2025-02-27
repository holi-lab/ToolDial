import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def scheduler_standalone_events(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns standalone events.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/scheduler/standalone-events"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scheduler_all_events_by_start_date(startdate: str='startDate', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all events for given day.
		"
    startdate: Specify start date of event.

        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/scheduler/all-events/by-start-date"
    querystring = {}
    if startdate:
        querystring['startDate'] = startdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scheduler_all_events(enddate: str='endDate', startdate: str='startDate', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all events in given window.
		"
    enddate: Specify end date of event.

        startdate: Specify start date of event.

        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/scheduler/all-events"
    querystring = {}
    if enddate:
        querystring['endDate'] = enddate
    if startdate:
        querystring['startDate'] = startdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_pods_id_relations(is_id: str, blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns relations for given block pod ID.
		"
    blockid: Specify Block ID
        keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/block-pods/{is_id}/relations"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocks_id_attachments(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns attachments associated with the given block.
		"
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/blocks/{is_id}/attachments"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def classroom_blocks_id_students_grades(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all students' teacher block grades for a given block.
		"
    keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/classroom-blocks/{is_id}/students/grades"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pod_types_id_pods(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns pods using given pod type.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/pod-types/{is_id}/pods"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def classroom_pods_id_submissions_attachments_as_teacher(is_id: str, keyid: str='63ceb29edb035900138d975d', blockid: str='93bwn23fje782486247d248h', studentid: str='84ytd25kdb035909838d992k', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns student attachment submissions (as teacher).
		"
    keyid: Specify Key ID.
        blockid: Specify Block ID.
        studentid: Specify student ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/classroom-pods/{is_id}/submissions/attachments/as-teacher"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    if studentid:
        querystring['studentId'] = studentid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pod_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns pod types. Set `includeCounts` to _true_ if you want a count of associated pods.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/pod-types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pods_archived(batchindex: str='batchIndex', keyid: str='67mnw82huw218472984b398h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns archived pods.
		"
    batchindex: Specify index of result set (aka, page number).
        keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/pods/archived"
    querystring = {}
    if batchindex:
        querystring['batchIndex'] = batchindex
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns app status.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/app/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def templates_blocks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block templates.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/templates/blocks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_relations(currentkeyid: str='63ceb29edb035900138d975d', currentblockid: str='93bwn23fje782486247d248h', currentpodid: str='63ceb29edb035900138d975d', token: str='token', blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns relations matching search token.
		"
    currentkeyid: Specify current key ID.

        currentblockid: Specify current block ID.

        currentpodid: Specify current pod ID.

        token: Specify search token.

        blockid: Specify Block ID (only needed for Block Pods)
        keyid: Specify Key ID (only needed for Block Pods)
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/search/relations"
    querystring = {}
    if currentkeyid:
        querystring['currentKeyId'] = currentkeyid
    if currentblockid:
        querystring['currentBlockId'] = currentblockid
    if currentpodid:
        querystring['currentPodId'] = currentpodid
    if token:
        querystring['token'] = token
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keys_id_pods(is_id: str, batchindex: str='batchIndex', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pods. If `batchIndex` is not specified, all pods will be returned in one go.
		"
    batchindex: Specify index of result set (aka, page number).
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/keys/{is_id}/pods"
    querystring = {}
    if batchindex:
        querystring['batchIndex'] = batchindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keys_id_pods_available_to_link(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pods available to be linked to this key. A pod can only be linked once to a key but it can be linked to any number of keys, and/or blocks.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/keys/{is_id}/pods/available-to-link"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keys(batchindex: str='batchIndex', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all keys. A key is the starting point, and is private to a user. Think of a key as a project, or a board.
		"
    batchindex: Specify index of result set (aka, page number).
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/keys"
    querystring = {}
    if batchindex:
        querystring['batchIndex'] = batchindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_keys_id_block_types(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all block types along with blocks in given key that use those block types.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/keys/{is_id}/block-types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def users(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all users.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/users"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_keys_id_blocks_block_id_linked_resources(block_id: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get block pods linked to given block or key.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/keys/{is_id}/blocks/{block_id}/linked-resources"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_pods_id(is_id: str, blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod. Alternate endpoint - `blocks/:id/pods/:id`.
		"
    blockid: Specify Block ID.
        keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/block-pods/{is_id}"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scales_id_blocks(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns blocks using given scale.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/scales/{is_id}/blocks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_dashboard_block_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all block types along with blocks that use those block types.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/dashboard/block-types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_dashboard_pod_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all pod types along with pods that use those pod types.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/dashboard/pod-types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_dashboard_keys_filters(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns `created by me`, `shared with me`, and `shared with others` filter results for all keys except system keys.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/dashboard/keys/filters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_dashboard_system_keys_filters(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns `created by me`, `shared with me`, and `shared with others` filter results for all system keys.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/dashboard/system-keys/filters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pods_id_checklists(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod checklists.
		"
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/pods/{is_id}/checklists"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_dashboard_task_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a breakdown of tasks (by status) across all keys.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/dashboard/task-status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def classroom_pods_id_students_grades(is_id: str, keyid: str='63ceb29edb035900138d975d', blockid: str='93bwn23fje782486247d248h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all students' teacher block grades for a given pod.
		"
    keyid: Specify Key ID.
        blockid: Specify Block ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/classroom-pods/{is_id}/students/grades"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_pods_id_attachments(is_id: str, blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns pod attachments.
		"
    blockid: Specify Block ID
        keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/block-pods/{is_id}/attachments"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def templates_pods(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns pod templates.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/templates/pods"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocks_id_project_block_lists(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns project lists.
		"
    keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/blocks/{is_id}/project-block-lists"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocks_id_students_all_grades(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns grades of all students for given block.
		"
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/blocks/{is_id}/students/all/grades"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_pods_archived(blockid: str='93bwn23fje782486247d248h', batchindex: str='batchIndex', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns archived block pods for given block.
		"
    blockid: Specify Block ID.
        batchindex: Specify index of result set (aka, page number).
        keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/block-pods/archived"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if batchindex:
        querystring['batchIndex'] = batchindex
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocks_id_pods(is_id: str, batchindex: str='batchIndex', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "If you want to fetch block pods in batches (_of 16_), set `batchIndex` to _true_. Else, all block pods will be returned in one go.
		"
    batchindex: Specify index of result set (aka, page number).
        keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/blocks/{is_id}/pods"
    querystring = {}
    if batchindex:
        querystring['batchIndex'] = batchindex
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dashboard_due_shortly_pods_and_tasks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns pods and tasks due shortly.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/dashboard/due-shortly/pods-and-tasks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profiles_introduction(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns user's introduction profile. This returns attributes that are pertinent to user's sign up, and typically used following a recent sign up.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/profiles/introduction"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dashboard_recently_modified(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns recently modified blocks and pods.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/dashboard/recently-modified"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_resource_attributes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns displayable attributes of Key, Block and Pod.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/app/resource/attributes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def profiles(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns user's profile.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/profiles"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def favorites(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all favorites - keys, blocks, key pods and block pods.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/favorites"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_types_id_blocks(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns blocks using given block type.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/block-types/{is_id}/blocks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keys_id_notes(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key notes.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/keys/{is_id}/notes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def notifications_unread(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns unread notifications.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/notifications/unread"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_keys_id_linked_resources(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns blocks & key pods linked to keys, and key pods linked to blocks.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/keys/{is_id}/linked-resources"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocks_id_linked_to_keys(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns keys a given block is linked to.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/blocks/{is_id}/linked-to/keys"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keys_filtered_by_type(keytype: str='keyType', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns keys filtered by given type, as defined in `keyType`.
		"
    keytype: Specify type of key.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/keys/filtered/by-type"
    querystring = {}
    if keytype:
        querystring['keyType'] = keytype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_blocks_id_shareable_users(is_id: str, token: str='token', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A block can be shared with a user only once. So, when you search for a user, this will return all users that match your search query that do not already have access to the block.
		"
    token: Specify search token.

        keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/search/blocks/{is_id}/shareable/users"
    querystring = {}
    if token:
        querystring['token'] = token
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_users(token: str='token', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches user by token.
		"
    token: Specify a search token.

        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/search/users"
    querystring = {}
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keys_archived(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns archived keys. Keys can be unarchived.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/keys/archived"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scales_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a scale.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/scales/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def classroom_students_id_profile(is_id: str, keyid: str='63ceb29edb035900138d975d', blockid: str='98nbc29nbr835753138d370c', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns student profile. This is specific to a Teacher Key.
		"
    keyid: Specify Key ID.

        blockid: Specify Block ID.

        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/classroom/students/{is_id}/profile"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_block_pods_id_shareable_users(is_id: str, blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', token: str='token', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A block pod can be shared with a user only once. So, when you search for a user, this will return all users that match your search query that do not already have access to the block pod.
		"
    blockid: Specify Block ID
        keyid: Specify Key ID
        token: Specify search token.

        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/search/block-pods/{is_id}/shareable/users"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pods_id_attachments(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod attachments."
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/pods/{is_id}/attachments"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_pods_id_comments(is_id: str, blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod comments.
		"
    blockid: Specify Block ID.
        keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/block-pods/{is_id}/comments"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def classroom_blocks_id_student_grades_as_teacher(is_id: str, keyid: str='63ceb29edb035900138d975d', studentuserid: str='84ytd25kdb035909838d992k', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block and pods' grades for a student. This request should be made by a teacher.
		"
    keyid: Specify Key ID.
        studentuserid: Specify student ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/classroom-blocks/{is_id}/student-grades/as-teacher"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if studentuserid:
        querystring['studentUserId'] = studentuserid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def classroom_pods_id_submissions_comments_as_teacher(is_id: str, studentid: str='84ytd25kdb035909838d992k', blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns student comment submissions (as teacher).
		"
    studentid: Specify student ID.
        blockid: Specify Block ID.
        keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/classroom-pods/{is_id}/submissions/comments/as-teacher"
    querystring = {}
    if studentid:
        querystring['studentId'] = studentid
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_dashboard_system_keys(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns system keys, blocks and pods.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/dashboard/system-keys"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_pods_id_acl(is_id: str, blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod collaborators. Alternate endpoint - `/blocks/:id/pods/:id/acl`.
		"
    blockid: Specify Block ID
        keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/block-pods/{is_id}/acl"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def templates_keys(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key templates.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/templates/keys"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block types. Set `includeCounts` to _true_ if you want a count of associated blocks.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/block-types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pods_id_tasks(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod tasks.
		"
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/pods/{is_id}/tasks"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pods_id(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod.
		"
    keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/pods/{is_id}"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def app_latest_version(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns latest version.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/app/latest-version"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def comments_recent(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns recent comments.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/comments/recent"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keys_id_relations(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns relations for given key ID.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/keys/{is_id}/relations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocks_id_project_block_lists_list_id(list_id: str, is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns project list for given ID.
		"
    keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/blocks/{is_id}/project-block-lists/{list_id}"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(token: str='token', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Searches key, block or pod by token.
		"
    token: Specify a search token.

        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/search"
    querystring = {}
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def classroom_pods_id_submissions_comments_as_student(is_id: str, blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns comment submissions (as student).
		"
    blockid: Specify Block ID.
        keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/classroom-pods/{is_id}/submissions/comments/as-student"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def classroom_blocks_id_students(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all students in a block.
		"
    keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/classroom-blocks/{is_id}/students"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def notifications_unread_count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns unread notification count.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/notifications/unread/count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def notifications(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all notifications.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/notifications"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_block_pods_id_tasks(is_id: str, keyid: str='63ceb29edb035900138d975d', blockid: str='93bwn23fje782486247d248h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod tasks.
		"
    keyid: Specify Key ID
        blockid: Specify Block ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/block-pods/{is_id}/tasks"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keys_id_checklists(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key checklists.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/keys/{is_id}/checklists"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pods_id_comments(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod comments. A comment is available to all collaborators.
		"
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/pods/{is_id}/comments"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pods_id_notes(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod notes. A note is private to the user.
		"
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/pods/{is_id}/notes"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_pods_id_notes(is_id: str, keyid: str='63ceb29edb035900138d975d', blockid: str='93bwn23fje782486247d248h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod notes. A note is private to a user, and is therefore not available to other collaborators.
		"
    keyid: Specify Key ID
        blockid: Specify Block ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/block-pods/{is_id}/notes"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocks_id_comments(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block comments. If the block is shared with other collaborators, the comment will be available to them as well.
		"
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/blocks/{is_id}/comments"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dashboard_recently_modified_keys(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns recently modified keys.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/dashboard/recently-modified/keys"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dashboard_due_shortly_blocks(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns blocks due shortly.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/dashboard/due-shortly/blocks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dashboard_unread_count(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns unread count for -
		  - Notifications 
		  - Conversations
		  - Blocks Due
		  - Pods Due
		  - Tasks Due
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/dashboard/unread-count"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dashboard_notifications_unread_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns unread notifications.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/dashboard/notifications/unread-status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dashboard_combined_responses(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the following dashboard details -
		  - Recently modified blocks
		  - Recently modified pods
		  - Pods due
		  - Tasks due
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/dashboard/combined-responses"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dashboard_conversations_unread_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns unread conversations.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/dashboard/conversations/unread-status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_keys_id_blocks_block_id_scales_scale_id_grades(scale_id: str, is_id: str, block_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block scale values for associated scale.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/keys/{is_id}/blocks/{block_id}/scales/{scale_id}/grades"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocks_id_pods_available_to_link(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns pods available to be linked to block. A pod can be linked to any number of blocks but a given pod can only be linked once to the same block.
		"
    keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/blocks/{is_id}/pods/available-to-link"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_keys_id_blocks_block_id_task_status(is_id: str, block_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns task status for given block.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/keys/{is_id}/blocks/{block_id}/task-status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_keys_id_scales(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all scales along with both blocks and pods in given key that use those scales.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/keys/{is_id}/scales"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_keys_id_scales_scale_id_scale_values(is_id: str, scale_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod and block scale values using given scale.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/keys/{is_id}/scales/{scale_id}/scale-values"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_keys_id_task_status(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a breakdown of tasks in given key.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/keys/{is_id}/task-status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keys_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key for given ID.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/keys/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_pods_id_checklists(is_id: str, blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod checklists. A block pod can have any number of checklists.
		"
    blockid: Specify Block ID
        keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/block-pods/{is_id}/checklists"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def classroom_pods_id_submissions_attachments_as_student(is_id: str, keyid: str='63ceb29edb035900138d975d', blockid: str='93bwn23fje782486247d248h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns attachment submissions (as student)
		"
    keyid: Specify Key ID.
        blockid: Specify Block ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/classroom-pods/{is_id}/submissions/attachments/as-student"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocks_id_acl(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all users who have access to a given block. Depending on where the block is located, there will be differences in access levels.
		- A block in a Teacher Key can be granted _admin_ or _read_ access. By granting admin access, you make that user a _Teacher_. If you granted _read_ access, the user will become a _Student_.
		- A block in a Project Key can be granted _admin_, _write_, or _read_ access. If you want the collaborator to be able to share the block in turn, grant them _admin_ access. Else, you can grant them _write_, or _read_.
		"
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/blocks/{is_id}/acl"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_classroom_pods_id_students_grades(is_id: str, keyid: str='63ceb29edb035900138d975d', blockid: str='93bwn23fje782486247d248h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod grades for all students
		"
    keyid: Specify Key ID
        blockid: Specify Block ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/classroom-pods/{is_id}/students/grades"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pods_id_acl(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key pod collaborators. Alternate endpoint - `/pods/:id/collaborators`.
		"
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/pods/{is_id}/acl"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocks_archived(keyid: str='63ceb29edb035900138d975d', batchindex: str='batchIndex', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of all archived blocks in given key.
		"
    keyid: Specify Key ID.
        batchindex: Specify index of result set (aka, page number).
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/blocks/archived"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if batchindex:
        querystring['batchIndex'] = batchindex
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_pods_id_shareable_users(is_id: str, token: str='token', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A key pod can be shared with a user only once. So, when you search for a user, this will return all users that match your search query that do not already have access to the key pod.
		"
    token: Specify search token.

        keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/search/pods/{is_id}/shareable/users"
    querystring = {}
    if token:
        querystring['token'] = token
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_keys_id_filters(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns `created by me`, `shared with me`, and `shared with others` filter results for given key.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/keys/{is_id}/filters"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocks_id(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block for given ID.
		"
    keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/blocks/{is_id}"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conversations(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns user conversations.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/conversations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def users_uuid_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns user by UUID.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/users/uuid/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocks_id_checklists(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block checklists.
		"
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/blocks/{is_id}/checklists"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def block_pods_id_tasks(is_id: str, blockid: str='93bwn23fje782486247d248h', keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block pod tasks.
		"
    blockid: Specify Block ID
        keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/block-pods/{is_id}/tasks"
    querystring = {}
    if blockid:
        querystring['blockId'] = blockid
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keys_id_blocks(is_id: str, batchindex: str='batchIndex', aclwriteorhigher: bool=None, filter: str='filter', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all blocks in a key.
		"
    batchindex: Specify index of result set (aka, page number).
        aclwriteorhigher: - True, if ACL is write or higher.
- False, otherwise.

        filter: Specify filter.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/keys/{is_id}/blocks"
    querystring = {}
    if batchindex:
        querystring['batchIndex'] = batchindex
    if aclwriteorhigher:
        querystring['aclWriteOrHigher'] = aclwriteorhigher
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conversations_unread_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns count of unread conversations (aka, conversations that have at least one message that user has not yet read).
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/conversations/unread-status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_keys_id_blocks_pods(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns blocks and pods associated with given key.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/keys/{is_id}/blocks-pods"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conversations_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns existing conversation for given ID.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/conversations/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocks_id_notes(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block note. A note is private to you and therefore, will not be available to your collaborators even if the block is shared.
		"
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/blocks/{is_id}/notes"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def conversations_by_usernames(usernames: str='user1,user2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns conversation for given usernames (if there happens to be one).
		"
    usernames: Comma-separated list of recipients' user names.

        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/conversations/by-usernames"
    querystring = {}
    if usernames:
        querystring['userNames'] = usernames
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_dashboard_scales(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all scales along with both blocks and pods that use those scales.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/dashboard/scales"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_keys_id_pod_types(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all pod types along with pods in given key that use those pod types.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/keys/{is_id}/pod-types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def charts_dashboard_keys_blocks_pods(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns user keys, blocks and pods.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/charts/dashboard/keys-blocks-pods"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scales(includecounts: str=None, excludeempty: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all scales.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/scales"
    querystring = {}
    if includecounts:
        querystring['includeCounts'] = includecounts
    if excludeempty:
        querystring['excludeEmpty'] = excludeempty
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def scales_id_pods(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns pods using given scale.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/scales/{is_id}/pods"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocks_id_tasks(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns block tasks.
		"
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/blocks/{is_id}/tasks"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keys_id_blocks_available_to_link(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A block can be linked to any number of keys but it can only be linked to a key once. Use this endpoint to get a list of all blocks that are available to be linked to a given key.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/keys/{is_id}/blocks/available-to-link"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pods_id_linked_to_blocks(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A pod can be linked to one or more blocks. This endpoint returns all blocks a given pod is linked to.
		"
    keyid: Specify Key ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/pods/{is_id}/linked-to/blocks"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pods_id_linked_to_keys(is_id: str, keyid: str='83yuz28ksn823022876m107v', blockid: str='93bwn23fje782486247d248h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns keys a given pod is linked to.
		"
    keyid: Specify Key ID.
        blockid: Specify Block ID.
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/pods/{is_id}/linked-to/keys"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    if blockid:
        querystring['blockId'] = blockid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pods_id_relations(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns relations for given pod ID.
		"
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/pods/{is_id}/relations"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blocks_id_relations(is_id: str, keyid: str='63ceb29edb035900138d975d', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns relations for given block ID.
		"
    keyid: Specify Key ID
        
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/blocks/{is_id}/relations"
    querystring = {}
    if keyid:
        querystring['keyId'] = keyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keys_id_tasks(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns key tasks.
		"
    
    """
    url = f"https://snowpal-project-management2.p.rapidapi.com/keys/{is_id}/tasks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "snowpal-project-management2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

