import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def user(email: str=None, lowercaseemail: str=None, lastseentimestamp: str=None, isoauthcomplete: str=None, displayname: str='"Joe Brown"', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User call"
    email: The email address of the user
        lowercaseemail: A lower case version of the users email address. Useful for normalization
        lastseentimestamp: The date the user last logged ino Streak
        isoauthcomplete: Whether the user has completed the OAuth process. Useful to determine whether they can successully share emails
        displayname: A display friendly name, usually the users full name if it exists in their profile
        
    """
    url = f"https://streak.p.rapidapi.com/user/me"
    querystring = {}
    if email:
        querystring['email'] = email
    if lowercaseemail:
        querystring['lowercaseEmail'] = lowercaseemail
    if lastseentimestamp:
        querystring['lastSeenTimestamp'] = lastseentimestamp
    if isoauthcomplete:
        querystring['isOauthComplete'] = isoauthcomplete
    if displayname:
        querystring['displayName'] = displayname
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streak.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pipeline(name: str=None, creatorkey: str=None, description: str=None, orgwide: str=None, fields: str=None, stages: str=None, stageorder: str=None, aclentries: str=None, owner: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pipelines represent a business process that a user would like managed."
    name: the name of this pipeline
        creatorkey: The user key of the user that created the pipeline
        description: the purpose of this pipeline, displayed in the web UI.
        orgwide: Whether this pipeline is shared with all users in the organization (same domain in email address)
        fields: what fields each box within the pipeline can have. This field is read-only. To modify, refer to the Fields endpoints
        stages: A map describing the set of possible stages a box within the pipeline can be in. Read-only and can only be modified using Stages endpoints
        stageorder: Editable array which allows you to reorder the stages. This modifies the order of the stages that appear in the web UI
        aclentries: An array of ACL objects (with properties: fullName, email, isOwner, image) which determines a list of users who have access to this pipeline
        owner: An object with the same properties as elements in the aclEntries array specifying the creator of this pipeline
        
    """
    url = f"https://streak.p.rapidapi.com/pipelines"
    querystring = {}
    if name:
        querystring['name'] = name
    if creatorkey:
        querystring['creatorKey'] = creatorkey
    if description:
        querystring['description'] = description
    if orgwide:
        querystring['orgWide'] = orgwide
    if fields:
        querystring['fields'] = fields
    if stages:
        querystring['stages'] = stages
    if stageorder:
        querystring['stageOrder'] = stageorder
    if aclentries:
        querystring['aclEntries'] = aclentries
    if owner:
        querystring['owner'] = owner
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "streak.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

