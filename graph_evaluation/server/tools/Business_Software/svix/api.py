import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_messages_api_v1_app_app_id_msg_get(app_id: str, limit: int=50, iterator: str='msg_1srOrx2ZWZBpBUvZwXKQmoEYga2', event_types: str='[
  "user.signup"
]', channel: str='project_1337', before: str=None, after: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all of the application's messages.
		
		The `before` parameter lets you filter all items created before a certain date and is ignored if an iterator is passed.
		The `after` parameter lets you filter all items created after a certain date and is ignored if an iterator is passed.
		`before` and `after` cannot be used simultaneously."
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/msg/"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if iterator:
        querystring['iterator'] = iterator
    if event_types:
        querystring['event_types'] = event_types
    if channel:
        querystring['channel'] = channel
    if before:
        querystring['before'] = before
    if after:
        querystring['after'] = after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_application_api_v1_app_app_id_get(app_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an application."
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_integration_key_api_v1_app_app_id_integration_integ_id_key_get(integ_id: str, app_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an integration's key."
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/integration/{integ_id}/key/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_attempt_api_v1_app_app_id_msg_msg_id_attempt_attempt_id_get(app_id: str, msg_id: str, attempt_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`msg_id`: Use a message id or a message `eventId`"
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/msg/{msg_id}/attempt/{attempt_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_attempts_by_msg_api_v1_app_app_id_attempt_msg_msg_id_get(app_id: str, msg_id: str, iterator: str='atmpt_1srOrx2ZWZBpBUvZwXKQmoEYga2', status: int=None, before: str=None, channel: str='project_1337', event_types: str='[
  "user.signup"
]', endpoint_id: str='ep_1srOrx2ZWZBpBUvZwXKQmoEYga2', status_code_class: int=None, limit: int=50, after: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List attempts by message id"
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/attempt/msg/{msg_id}/"
    querystring = {}
    if iterator:
        querystring['iterator'] = iterator
    if status:
        querystring['status'] = status
    if before:
        querystring['before'] = before
    if channel:
        querystring['channel'] = channel
    if event_types:
        querystring['event_types'] = event_types
    if endpoint_id:
        querystring['endpoint_id'] = endpoint_id
    if status_code_class:
        querystring['status_code_class'] = status_code_class
    if limit:
        querystring['limit'] = limit
    if after:
        querystring['after'] = after
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_attempts_for_endpoint_api_v1_app_app_id_msg_msg_id_endpoint_endpoint_id_attempt_get(msg_id: str, app_id: str, endpoint_id: str, iterator: str='atmpt_1srOrx2ZWZBpBUvZwXKQmoEYga2', event_types: str='[
  "user.signup"
]', before: str=None, limit: int=50, status: int=None, after: str=None, channel: str='project_1337', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "DEPRECATED: please use list_attempts with endpoint_id as a query parameter instead.
		
		List the message attempts for a particular endpoint.
		
		Returning the endpoint.
		
		The `before` parameter lets you filter all items created before a certain date and is ignored if an iterator is passed."
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/msg/{msg_id}/endpoint/{endpoint_id}/attempt/"
    querystring = {}
    if iterator:
        querystring['iterator'] = iterator
    if event_types:
        querystring['event_types'] = event_types
    if before:
        querystring['before'] = before
    if limit:
        querystring['limit'] = limit
    if status:
        querystring['status'] = status
    if after:
        querystring['after'] = after
    if channel:
        querystring['channel'] = channel
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_message_api_v1_app_app_id_msg_msg_id_get(msg_id: str, app_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a message by its ID or eventID."
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/msg/{msg_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_attempted_messages_api_v1_app_app_id_endpoint_endpoint_id_msg_get(app_id: str, endpoint_id: str, after: str=None, iterator: str='msg_1srOrx2ZWZBpBUvZwXKQmoEYga2', before: str=None, limit: int=50, channel: str='project_1337', status: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List messages for a particular endpoint. Additionally includes metadata about the latest message attempt.
		
		The `before` parameter lets you filter all items created before a certain date and is ignored if an iterator is passed."
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/endpoint/{endpoint_id}/msg/"
    querystring = {}
    if after:
        querystring['after'] = after
    if iterator:
        querystring['iterator'] = iterator
    if before:
        querystring['before'] = before
    if limit:
        querystring['limit'] = limit
    if channel:
        querystring['channel'] = channel
    if status:
        querystring['status'] = status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_attempted_destinations_api_v1_app_app_id_msg_msg_id_endpoint_get(app_id: str, msg_id: str, iterator: str='msgep_1srOrx2ZWZBpBUvZwXKQmoEYga2', limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`msg_id`: Use a message id or a message `eventId`"
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/msg/{msg_id}/endpoint/"
    querystring = {}
    if iterator:
        querystring['iterator'] = iterator
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_integration_api_v1_app_app_id_integration_integ_id_get(integ_id: str, app_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an integration."
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/integration/{integ_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_attempts_api_v1_app_app_id_msg_msg_id_attempt_get(app_id: str, msg_id: str, status: int=None, channel: str='project_1337', event_types: str='[
  "user.signup"
]', after: str=None, limit: int=50, iterator: str='atmpt_1srOrx2ZWZBpBUvZwXKQmoEYga2', endpoint_id: str='ep_1srOrx2ZWZBpBUvZwXKQmoEYga2', before: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Deprecated: Please use "List Attempts by Endpoint" and "List Attempts by Msg" instead.
		
		`msg_id`: Use a message id or a message `eventId`"
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/msg/{msg_id}/attempt/"
    querystring = {}
    if status:
        querystring['status'] = status
    if channel:
        querystring['channel'] = channel
    if event_types:
        querystring['event_types'] = event_types
    if after:
        querystring['after'] = after
    if limit:
        querystring['limit'] = limit
    if iterator:
        querystring['iterator'] = iterator
    if endpoint_id:
        querystring['endpoint_id'] = endpoint_id
    if before:
        querystring['before'] = before
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_endpoint_secret_api_v1_app_app_id_endpoint_endpoint_id_secret_get(endpoint_id: str, app_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the endpoint's signing secret.
		
		This is used to verify the authenticity of the webhook.
		For more information please refer to [the consuming webhooks docs](https://docs.svix.com/consuming-webhooks/)."
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/endpoint/{endpoint_id}/secret/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_attempts_by_endpoint_api_v1_app_app_id_attempt_endpoint_endpoint_id_get(app_id: str, endpoint_id: str, limit: int=50, event_types: str='[
  "user.signup"
]', after: str=None, channel: str='project_1337', before: str=None, status_code_class: int=None, status: int=None, iterator: str='atmpt_1srOrx2ZWZBpBUvZwXKQmoEYga2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List attempts by endpoint id"
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/attempt/endpoint/{endpoint_id}/"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if event_types:
        querystring['event_types'] = event_types
    if after:
        querystring['after'] = after
    if channel:
        querystring['channel'] = channel
    if before:
        querystring['before'] = before
    if status_code_class:
        querystring['status_code_class'] = status_code_class
    if status:
        querystring['status'] = status
    if iterator:
        querystring['iterator'] = iterator
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def health_api_v1_health_get(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verify the API server is up and running."
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/health/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_endpoint_api_v1_app_app_id_endpoint_endpoint_id_get(app_id: str, endpoint_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an application."
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/endpoint/{endpoint_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_event_type_api_v1_event_type_event_type_name_get(event_type_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an event type."
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/event-type/{event_type_name}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_integrations_api_v1_app_app_id_integration_get(app_id: str, iterator: str='integ_1srOrx2ZWZBpBUvZwXKQmoEYga2', limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the application's integrations."
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/integration/"
    querystring = {}
    if iterator:
        querystring['iterator'] = iterator
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_endpoint_headers_api_v1_app_app_id_endpoint_endpoint_id_headers_get(endpoint_id: str, app_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the additional headers to be sent with the webhook"
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/endpoint/{endpoint_id}/headers/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_event_types_api_v1_event_type_get(iterator: str='user.signup', with_content: bool=None, limit: int=50, include_archived: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the list of event types."
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/event-type/"
    querystring = {}
    if iterator:
        querystring['iterator'] = iterator
    if with_content:
        querystring['with_content'] = with_content
    if limit:
        querystring['limit'] = limit
    if include_archived:
        querystring['include_archived'] = include_archived
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_endpoints_api_v1_app_app_id_endpoint_get(app_id: str, iterator: str='ep_1srOrx2ZWZBpBUvZwXKQmoEYga2', limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the application's endpoints."
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/{app_id}/endpoint/"
    querystring = {}
    if iterator:
        querystring['iterator'] = iterator
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_applications_api_v1_app_get(iterator: str='app_1srOrx2ZWZBpBUvZwXKQmoEYga2', limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all the organization's applications."
    
    """
    url = f"https://svix1.p.rapidapi.com/api/v1/app/"
    querystring = {}
    if iterator:
        querystring['iterator'] = iterator
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "svix1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

