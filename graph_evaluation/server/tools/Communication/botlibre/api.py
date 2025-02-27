import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def form_check_instance(password: str, is_id: str='1234', name: str='mybot', application: str='myapp', user: str='myuserid', token: str='123456', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The form-check-instance API validates that a bot ID or name exists, and returns the bot's details."
    password: OPTIONAL: The password of the user. A token can also be used.
        is_id: REQUIRED: The ID of the bot to validate. The bot's name can also be used, but the ID is better as it is guaranteed to be unique.
        name: REQUIRED: The nameof the bot to validate. The bot's ID can also be used.
        application: OPTIONAL: The application ID. If not passed, the application will be anonymous.
        user: OPTIONAL: The ID of the user. The user must be registered with BOT libre. If not passed the user will be anonymous. The user is required if the bot is private.
        token: OPTIONAL: The token of the user. A token can be obtained through check-user, and is valid until reset.
        
    """
    url = f"https://paphus-botlibre.p.rapidapi.com/form-check-instance"
    querystring = {'password': password, }
    if is_id:
        querystring['id'] = is_id
    if name:
        querystring['name'] = name
    if application:
        querystring['application'] = application
    if user:
        querystring['user'] = user
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "paphus-botlibre.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def form_check_user(user: str, application: str='myapp', password: str='password', token: str='123456', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The form-check-user API validates user, and returns the user's details."
    user: REQUIRED: The ID of the user. The user must be registered with BOT libre. If not passed the user will be anonymous. The user is required if the bot is private.
        application: OPTIONAL: The application ID. If not passed, the application will be anonymous.
        password: REQUIRED: The password of the user. A token can also be used.
        token: REQUIRED: The token of the user. A token can be obtained through check-user, and is valid until reset.
        
    """
    url = f"https://paphus-botlibre.p.rapidapi.com/form-check-user"
    querystring = {'user': user, }
    if application:
        querystring['application'] = application
    if password:
        querystring['password'] = password
    if token:
        querystring['token'] = token
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "paphus-botlibre.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def form_chat(instance: str, application: str='myapp', user: str='myuserid', password: str='mypassword', token: str='123456', conversation: str='1234', message: str='Hello  Bot', emote: str='HAPPY', correction: str=None, offensive: str=None, disconnect: str=None, includequestion: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The form-chat API receives a chat message and returns the chat bot's reply as an XML document."
    instance: REQUIRED: The ID of the bot to chat with. The bot's name can also be used, but the ID is better as it is guaranteed to be unique.
        application: OPTIONAL: The application ID. If not passed, the application will be anonymous.
        user: OPTIONAL: The ID of the user who is sending the message. The user must be registered with BOT libre. If not passed the user will be anonymous. The user is required if the bot is private. The user/password are only required on the first message.
        password: OPTIONAL: The password of the user who is sending the message. A token can also be used.
        token: OPTIONAL: The token of the user who is sending the message. A token can be obtained through check-user, and is valid until reset.
        conversation: OPTIONAL: The conversation ID for the current conversation. This must not be passed on the first message, but will be returned by the response, and should be used for all subsequent messages in the conversation.
        message: OPTIONAL: The chat message to send to the bot. The message can be omitted if you wish the bot to start the conversation (if the bot has a greeting set). The message must be encoded in the URI.
        emote: OPTIONAL: A emotion to tag the message with. This is one of LOVE, LIKE, DISLIKE, HATE, RAGE, ANGER, CALM, SERENE, ECSTATIC, HAPPY, SAD, CRYING, PANIC, AFRAID, CONFIDENT, COURAGEOUS, SURPRISE, BORED, LAUGHTER, SERIOUS.
        correction: OPTIONAL: A boolean that defines the chat message is a correction to the bot's last answer.
        offensive: OPTIONAL: A boolean that defines the bot's last answer as offensive. The message will be flagged for the bot's administrator to review.
        disconnect: OPTIONAL: A boolean that defines the end of the conversation.
        includequestion: OPTIONAL: A boolean that indicates the question should be included in the response.
        
    """
    url = f"https://paphus-botlibre.p.rapidapi.com/form-chat"
    querystring = {'instance': instance, }
    if application:
        querystring['application'] = application
    if user:
        querystring['user'] = user
    if password:
        querystring['password'] = password
    if token:
        querystring['token'] = token
    if conversation:
        querystring['conversation'] = conversation
    if message:
        querystring['message'] = message
    if emote:
        querystring['emote'] = emote
    if correction:
        querystring['correction'] = correction
    if offensive:
        querystring['offensive'] = offensive
    if disconnect:
        querystring['disconnect'] = disconnect
    if includequestion:
        querystring['includeQuestion'] = includequestion
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "paphus-botlibre.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

