import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_all_following(userid: str, skip: int=0, take: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the set of entities followed by the given user."
    skip: Used for paging. The number of entities to skip before returning a set of entities. Defaults to 0.
        take: Used for paging. The number of entities to return. Defaults to 20.
        
    """
    url = f"https://collabinate.p.rapidapi.com/users/{userid}/following"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "collabinate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_comments(entityid: str, activityid: str, start: int=0, count: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the set of comments for a given activity."
    entityid: The ID of the entity which contains the activity for which to get the comments.
        activityid: The ID of the activity to which the comments belong.
        start: Used for paging. The number of comments to skip before returning a set of comments. Defaults to 0.
        count: Used for paging. The number of comments to return. Defaults to 20.
        
    """
    url = f"https://collabinate.p.rapidapi.com/entities/{entityId}/stream/{activityId}/comments"
    querystring = {'entityid': entityid, 'activityid': activityid, }
    if start:
        querystring['start'] = start
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "collabinate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_following(userid: str, entityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a follow activity if the given user follows the given entity, or a 404 not found otherwise."
    userid: The ID of the user to check.
        entityid: The ID of the entity to check.
        
    """
    url = f"https://collabinate.p.rapidapi.com/users/{userId}/following/{entityId}"
    querystring = {'userid': userid, 'entityid': entityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "collabinate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_activity(entityid: str, activityid: str, comments: int=20, likes: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single activity within a given stream."
    entityid: The ID of the entity from which's stream the activity will be retrieved.
        activityid: The ID of the activity to retrieve.
        comments: The number of comments on the activity to embed in the response. Note that setting this to any value (even 0) will embed a collection that contains a value for the count of all comments on the activity, along with a set of the most recent comment objects up to the number specified. For high volume usage, leave this parameter out of the query for best performance.
        likes: The number of likes on the activity to embed in the response. Note that setting this to any value (even 0) will embed a collection that contains a value for the count of all likes on the activity, along with a set of like objects up to the number specified. For high volume usage, leave this parameter out of the query for best performance.
        
    """
    url = f"https://collabinate.p.rapidapi.com/entities/{entityId}/stream/{activityId}"
    querystring = {'entityid': entityid, 'activityid': activityid, }
    if comments:
        querystring['comments'] = comments
    if likes:
        querystring['likes'] = likes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "collabinate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_likers(entityid: str, activityid: str, skip: int=0, take: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a set of users that like a given activity."
    entityid: The ID of the entity which has the liked activity in its stream.
        activityid: The ID of the activity for which to retrieve the liking users.
        skip: Used for paging. The number of users to skip before returning a set of users. Defaults to 0.
        take: Used for paging. The number of users to return. Defaults to 20.
        
    """
    url = f"https://collabinate.p.rapidapi.com/entities/{entityId}/stream/{activityId}/likes"
    querystring = {'entityid': entityid, 'activityid': activityid, }
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "collabinate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_followers(entityid: str, skip: int=0, take: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the set of users that follow the given entity."
    skip: Used for paging. The number of users to skip before returning a set of users. Defaults to 0.
        take: Used for paging. The number of users to return. Defaults to 20.
        
    """
    url = f"https://collabinate.p.rapidapi.com/entities/{entityid}/followers"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "collabinate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_comment(entityid: str, activityid: str, commentid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a single comment on an activity."
    entityid: The ID of the entity which contains the activity with the comment in its stream.
        activityid: The ID of the activity to which the comment belongs.
        commentid: The ID of the comment.
        
    """
    url = f"https://collabinate.p.rapidapi.com/entities/{entityId}/stream/{activityId}/comments/{commentId}"
    querystring = {'entityid': entityid, 'activityid': activityid, 'commentid': commentid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "collabinate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_entity(entityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets an ActivityStreamsObject representing an Entity or User."
    
    """
    url = f"https://collabinate.p.rapidapi.com/entities/{entityid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "collabinate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stream(entityid: str, skip: int=0, take: int=20, comments: int=20, likes: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a set of activities for an entity in reverse chronological order up to the amount in the take query variable, starting after the skipping the number of activities in the skip query variable.  If not specified, skip defaults to zero, and take defaults to 20."
    entityid: The ID of the entity for which the stream will be retrieved.
        skip: Used for paging. The number of activities to skip before returning a set of activities. Defaults to 0.
        take: Used for paging. The number of activities to return. Defaults to 20.
        comments: The number of comments on each activity to embed in the response. Note that setting this to any value (even 0) will embed a collection that contains a value for the count of all comments on each activity, along with a set of the most recent comment objects up to the number specified. For high volume usage, leave this parameter out of the query for best performance.
        likes: The number of likes on each activity to embed in the response. Note that setting this to any value (even 0) will embed a collection that contains a value for the count of all likes on each activity, along with a set of like objects up to the number specified. For high volume usage, leave this parameter out of the query for best performance.
        
    """
    url = f"https://collabinate.p.rapidapi.com/entities/{entityId}/stream"
    querystring = {'entityid': entityid, }
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    if comments:
        querystring['comments'] = comments
    if likes:
        querystring['likes'] = likes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "collabinate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_like(userid: str, entityid: str, activityid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Checks whether a like relationship exists between a user and a single activity."
    userid: The ID of the user to check.
        entityid: The ID of the entity which has the liked activity in its stream.
        activityid: The ID of the activity to check if the user likes.
        
    """
    url = f"https://collabinate.p.rapidapi.com/users/{userId}/likes/{entityId}/{activityId}"
    querystring = {'userid': userid, 'entityid': entityid, 'activityid': activityid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "collabinate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_feed(userid: str, skip: int=0, take: int=20, comments: int=20, likes: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a set of activities for the entities followed by a user in reverse chronological order up to the amount in the take query variable, starting at the activity in the position after the skip variable.  If not specified, skip defaults to zero, and take defaults to 20."
    userid: The ID of the user for which the feed will be retrieved.
        skip: Used for paging. The number of activities to skip before returning a set of activities. Defaults to 0.
        take: Used for paging. The number of activities to return. Defaults to 20.
        comments: The number of comments on each activity to embed in the response. Note that setting this to any value (even 0) will embed a collection that contains a value for the count of all comments on each activity, along with a set of the most recent comment objects up to the number specified. For high volume usage, leave this parameter out of the query for best performance.
        likes: The number of likes on each activity to embed in the response. Note that setting this to any value (even 0) will embed a collection that contains a value for the count of all likes on each activity, along with a set of like objects up to the number specified. For high volume usage, leave this parameter out of the query for best performance.
        
    """
    url = f"https://collabinate.p.rapidapi.com/users/{userId}/feed"
    querystring = {'userid': userid, }
    if skip:
        querystring['skip'] = skip
    if take:
        querystring['take'] = take
    if comments:
        querystring['comments'] = comments
    if likes:
        querystring['likes'] = likes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "collabinate.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

