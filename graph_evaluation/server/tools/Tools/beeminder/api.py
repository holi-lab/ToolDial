import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_information_about_a_user(username: str, goals_filter: str=None, associations: bool=None, diff_since: int=None, skinny: bool=None, datapoints_count: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves information and a list of goal names for the user with username u."
    username: Since appending an access_token to the request uniquely identifies a user, you can alternatively make the request to /users/me.json (without the username).
        goals_filter: One of {all, frontburner, backburner}. Default: all, which returns an unfiltered list of all goals. Send frontburner if you want the list of goals the user has marked as frontburner goals (those which appear above the fold in the web interface), or backburner for just the below-the-fold goals.
        associations: Convenience method to fetch all information about a user. Please use sparingly and see also the diff_since parameter. Default: false Send true if you want to receive all of the user’s goal and datapoints as attributes of the user object.
        diff_since: Unix timestamp in seconds. Default: null, which will return all goals and datapoints Send a Unix timestamp to receive a filtered list of the user’s goals and datapoints. Only goals and datapoints that have been created or updated since the timestamp will be returned. Sending diff_since implies that you want the user’s associations, so you don’t need to send both.
        skinny: Convenience method to only get a subset of goal attributes and the most recent datapoint for the goal. Default: false, which will return all goal attributes and all datapoints created or updated since diff_since. skinny must be sent along with diff_since. If diff_since is not present, skinny is ignored. Some goal attributes, as well as fetching all datapoints, can take some additional time to compute on the server side, so you can send skinny if you only need the latest datapoint and the following subset of attributes: slug, title, goalval, rate, goaldate, graph_url, thumb_url, goal_type, losedate, id, ephem, queued, panic, updated_at, burner, yaw, runits, lane, frozen, won, lost Instead of a datapoints attribute, sending skinny will replace that attribute with a last_datapoint attribute. Its value is a Datapoint hash.
        datapoints_count: number of datapoints. Default: null, which will return all goals and datapoints Send a number n to only recieve the n most recently added datapoints, sorted by updated_at. Note that the most recently added datapoint could have been a datapoint whose timestamp is well in the past and therefore before other datapoints in that respect.
        
    """
    url = f"https://dreeves-beeminder.p.rapidapi.com/users/{username}.json"
    querystring = {}
    if goals_filter:
        querystring['goals_filter'] = goals_filter
    if associations:
        querystring['associations'] = associations
    if diff_since:
        querystring['diff_since'] = diff_since
    if skinny:
        querystring['skinny'] = skinny
    if datapoints_count:
        querystring['datapoints_count'] = datapoints_count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dreeves-beeminder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_the_datapoints(u: str, g: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of datapoints for user u’s goal g — beeminder.com/u/g."
    u: user u
        g: goal g
        
    """
    url = f"https://dreeves-beeminder.p.rapidapi.com/users/{u}/goals/{g}/datapoints.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dreeves-beeminder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_information_about_a_goal(datapoints: bool, username: str, goal: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets goal details for user u’s goal g — beeminder.com/u/g."
    datapoints: Whether to to send the goal’s datapoints in the response. Default: false.
        username: Since appending an access_token to the request uniquely identifies a user, you can alternatively make the request to /users/me.json (without the username).
        goal: Goal of user
        
    """
    url = f"https://dreeves-beeminder.p.rapidapi.com/users/{username}/goals/{goal}.json"
    querystring = {'datapoints': datapoints, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dreeves-beeminder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_goals_for_a_user(filter: str, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get user u’s list of goals."
    filter: One of {all, frontburner, backburner}. Default: all.
        username: The Beeminder username of the user being charged.
        
    """
    url = f"https://dreeves-beeminder.p.rapidapi.com/users/{username}/goals.json"
    querystring = {'filter': filter, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dreeves-beeminder.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

