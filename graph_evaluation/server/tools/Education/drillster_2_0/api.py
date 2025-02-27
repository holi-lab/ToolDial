import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_account_access(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Requests an organization login."
    user_id: User ID
        
    """
    url = f"https://drillster2.p.rapidapi.com/access/{user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_drill_details(drill_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves drill or course details"
    drill_id: The drill or course ID
        
    """
    url = f"https://drillster2.p.rapidapi.com/drill/{drill_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_group_members(group_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves all members of a specific group."
    
    """
    url = f"https://drillster2.p.rapidapi.com/group/{group_id}/members"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_group_drills(group_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves all drills of a specific group. These can be regular drills, and courses. A course is a collection of (regular) drills."
    group_id: The unique ID of the group
        
    """
    url = f"https://drillster2.p.rapidapi.com/group/{group_id}/drills"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_group_objectives(group_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves all objectives that have been defined for a group."
    group_id: The unique ID for the group
        
    """
    url = f"https://drillster2.p.rapidapi.com/group/{group_id}/objectives"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_practice_question(drill_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the next question for the specified drill or collection of drills."
    drill_id: The unique ID of the drill
        
    """
    url = f"https://drillster2.p.rapidapi.com/question/{drill_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_own_test_result(test_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves an individual test result for a specific test instance for the calling user."
    test_id: The unique ID of the test definition
        
    """
    url = f"https://drillster2.p.rapidapi.com/test-result/{test_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_objectives(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the objectives that have been defined for the calling user."
    
    """
    url = f"https://drillster2.p.rapidapi.com/objectives"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_in_drill_store(query: str, searchfield: str='NAME', resultstart: str='0', resultsize: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search the Drill Store for available drills and courses."
    query: The search query
        searchfield: Specifies which fields will be searched. Can be included multiple times to specify multiple search fields. If not specified, all available fields will be searched.
        resultstart: The index number (zero-based) of the first drill (or course) to include in the result. Use this parameter, in combination with resultSize, to apply pagination of the result; it will limit the number of drills included in the result. Defaults to 0.
        resultsize: The maximum number of drills (or courses) to include in the result. If not specified, the results for all (remaining) drills are returned.
        
    """
    url = f"https://drillster2.p.rapidapi.com/store/{query}"
    querystring = {}
    if searchfield:
        querystring['searchField'] = searchfield
    if resultstart:
        querystring['resultStart'] = resultstart
    if resultsize:
        querystring['resultSize'] = resultsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_test_question(test_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the next question for a predefined test."
    test_id: The unique ID for the predefined test
        
    """
    url = f"https://drillster2.p.rapidapi.com/test-question/{test_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_own_user(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Requests (own) user details."
    
    """
    url = f"https://drillster2.p.rapidapi.com/user"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_practice_preferences(drill_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the user's practice preferences for a given drill."
    drill_id: The unique ID of the drill
        
    """
    url = f"https://drillster2.p.rapidapi.com/practice-preferences/{drill_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def submit_test_answer(reference: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Submits the answer to a question, as part of a predefined test."
    reference: The unique identifier of the answered question
        
    """
    url = f"https://drillster2.p.rapidapi.com/test-answer/{reference}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_group_results(group_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the up-to-date results for a group."
    group_id: The unique ID for the group
        
    """
    url = f"https://drillster2.p.rapidapi.com/group/{group_id}/results"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_list_of_groups(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves a list of all groups the calling user has access to."
    
    """
    url = f"https://drillster2.p.rapidapi.com/groups"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_user(user_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves (other) user details."
    user_id: User ID
        
    """
    url = f"https://drillster2.p.rapidapi.com/user/{user_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_in_repertoire(sortdirection: str, resultsize: str='10', query: str='chinese AND mandarin', searchfield: str='NAME', resultstart: str='0', sortfield: str='NAME', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the user's repertoire, or searches in it using a search query."
    sortdirection: The sort direction (ascending or descending), if a sortField has been specified. If omitted, a default sort direction is used, which depends on the sortField. Possible values are: ASC, DESC.
        resultsize: The maximum number of drills (or courses) to include in the result. If not specified, the results for all (remaining) drills are returned.
        query: The text to search for. It can be a single word, or multiple words (separated with a space). The search is case-insensitive. If multiple words are specified, the default search behavior is to search for any of the words. The AND operator can be used to search for all words, e.g. "milk AND butter". Note that the AND operator must be in capitals.
        searchfield: Specifies which fields will be searched. Can be included multiple times to specify multiple search fields. If not specified, all available fields will be searched. Possible values are:  NAME, DESCRIPTION, SUBJECT, TAGS, CONTENTS.
        resultstart: The index number (zero-based) of the first drill (or course) to include in the result. Use this parameter, in combination with resultSize, to apply pagination of the result; it will limit the number of drills included in the result. Defaults to 0.
        sortfield: The field to sort the result by.
        
    """
    url = f"https://drillster2.p.rapidapi.com/repertoire"
    querystring = {'sortDirection': sortdirection, }
    if resultsize:
        querystring['resultSize'] = resultsize
    if query:
        querystring['query'] = query
    if searchfield:
        querystring['searchField'] = searchfield
    if resultstart:
        querystring['resultStart'] = resultstart
    if sortfield:
        querystring['sortField'] = sortfield
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_test_result(test_id: str, user_id_or_ticket: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves an individual test result for a specific test instance."
    test_id: The unique ID for the test definition
        user_id_or_ticket: The user ID to retrieve the test results for, or in the case of an anonymous test, the test ticket
        
    """
    url = f"https://drillster2.p.rapidapi.com/test-result/{test_id}/{user_id_or_ticket}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "drillster2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

