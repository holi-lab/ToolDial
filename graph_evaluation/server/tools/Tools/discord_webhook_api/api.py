import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def repeat_send_message(repeat: int, webhook_url: str, message: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter your webhook URL, enter how many times to send the message, and then specify the message to send. Max 10 messages."
    
    """
    url = f"https://discord-webhook-api.p.rapidapi.com/repeat_send_message"
    querystring = {'repeat': repeat, 'webhook_url': webhook_url, 'message': message, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "discord-webhook-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def send_embed(content: str, webhook_url: str, field22title: str=None, field21content: str=None, field19title: str=None, field17content: str=None, field14content: str=None, field16title: str=None, field17title: str=None, field15content: str=None, field14title: str=None, field10title: str=None, field13content: str=None, field12title: str=None, field9content: str=None, field6content: str=None, field9title: str=None, field12content: str=None, field18content: str=None, field11title: str=None, field8content: str=None, field5content: str=None, field11content: str=None, field3title: str=None, field10content: str=None, field6title: str=None, field5title: str=None, field2title: str='Second field title', field25title: str=None, thumbnail: str='This is the URL to the thumbnail', field3content: str=None, field4title: str=None, field7title: str=None, field7content: str=None, field8title: str=None, field2content: str='Second field content', field_count: int=2, author_icon: str="This is the URL to the author's icon", author_url: str='This is the URL that you go to when you click the author_name', image: str='This is the URL to the image', author_name: str='This is the name of the author', title: str='This is the title of the embed', footer: str='This is a footer', timestamp: bool=None, field1content: str='First field content', footer_icon: str="This is the URL to the footer's icon", field1title: str='First field title', field4content: str=None, field20content: str=None, color: str='Hexadecimal color (no #)', field24title: str=None, inline: bool=None, field23title: str=None, field25content: str=None, field23content: str=None, field24content: str=None, field16content: str=None, field21title: str=None, field20title: str=None, field19content: str=None, field15title: str=None, field13title: str=None, field22content: str=None, field18title: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows you to send an embed via a webhook. Read the documentation to get this to work."
    
    """
    url = f"https://discord-webhook-api.p.rapidapi.com/send_embed"
    querystring = {'content': content, 'webhook_url': webhook_url, }
    if field22title:
        querystring['field22title'] = field22title
    if field21content:
        querystring['field21content'] = field21content
    if field19title:
        querystring['field19title'] = field19title
    if field17content:
        querystring['field17content'] = field17content
    if field14content:
        querystring['field14content'] = field14content
    if field16title:
        querystring['field16title'] = field16title
    if field17title:
        querystring['field17title'] = field17title
    if field15content:
        querystring['field15content'] = field15content
    if field14title:
        querystring['field14title'] = field14title
    if field10title:
        querystring['field10title'] = field10title
    if field13content:
        querystring['field13content'] = field13content
    if field12title:
        querystring['field12title'] = field12title
    if field9content:
        querystring['field9content'] = field9content
    if field6content:
        querystring['field6content'] = field6content
    if field9title:
        querystring['field9title'] = field9title
    if field12content:
        querystring['field12content'] = field12content
    if field18content:
        querystring['field18content'] = field18content
    if field11title:
        querystring['field11title'] = field11title
    if field8content:
        querystring['field8content'] = field8content
    if field5content:
        querystring['field5content'] = field5content
    if field11content:
        querystring['field11content'] = field11content
    if field3title:
        querystring['field3title'] = field3title
    if field10content:
        querystring['field10content'] = field10content
    if field6title:
        querystring['field6title'] = field6title
    if field5title:
        querystring['field5title'] = field5title
    if field2title:
        querystring['field2title'] = field2title
    if field25title:
        querystring['field25title'] = field25title
    if thumbnail:
        querystring['thumbnail'] = thumbnail
    if field3content:
        querystring['field3content'] = field3content
    if field4title:
        querystring['field4title'] = field4title
    if field7title:
        querystring['field7title'] = field7title
    if field7content:
        querystring['field7content'] = field7content
    if field8title:
        querystring['field8title'] = field8title
    if field2content:
        querystring['field2content'] = field2content
    if field_count:
        querystring['field_count'] = field_count
    if author_icon:
        querystring['author_icon'] = author_icon
    if author_url:
        querystring['author_url'] = author_url
    if image:
        querystring['image'] = image
    if author_name:
        querystring['author_name'] = author_name
    if title:
        querystring['title'] = title
    if footer:
        querystring['footer'] = footer
    if timestamp:
        querystring['timestamp'] = timestamp
    if field1content:
        querystring['field1content'] = field1content
    if footer_icon:
        querystring['footer_icon'] = footer_icon
    if field1title:
        querystring['field1title'] = field1title
    if field4content:
        querystring['field4content'] = field4content
    if field20content:
        querystring['field20content'] = field20content
    if color:
        querystring['color'] = color
    if field24title:
        querystring['field24title'] = field24title
    if inline:
        querystring['inline'] = inline
    if field23title:
        querystring['field23title'] = field23title
    if field25content:
        querystring['field25content'] = field25content
    if field23content:
        querystring['field23content'] = field23content
    if field24content:
        querystring['field24content'] = field24content
    if field16content:
        querystring['field16content'] = field16content
    if field21title:
        querystring['field21title'] = field21title
    if field20title:
        querystring['field20title'] = field20title
    if field19content:
        querystring['field19content'] = field19content
    if field15title:
        querystring['field15title'] = field15title
    if field13title:
        querystring['field13title'] = field13title
    if field22content:
        querystring['field22content'] = field22content
    if field18title:
        querystring['field18title'] = field18title
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "discord-webhook-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def multi_send_message(message_count: int, message1: str, webhook_url: str, message9: str=None, message4: str=None, message8: str=None, message7: str=None, message6: str=None, message3: str=None, message10: str=None, message2: str=None, message5: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows you to send multiple messages with a webhook. More information in the docs. Modify the message_count value to the amount of messages you wish to send, and then specify each of the contents of the messages. If you wish to send the same message several times, use repeat send message.
		Maximum 10 messages at once."
    
    """
    url = f"https://discord-webhook-api.p.rapidapi.com/multi_send_message"
    querystring = {'message_count': message_count, 'message1': message1, 'webhook_url': webhook_url, }
    if message9:
        querystring['message9'] = message9
    if message4:
        querystring['message4'] = message4
    if message8:
        querystring['message8'] = message8
    if message7:
        querystring['message7'] = message7
    if message6:
        querystring['message6'] = message6
    if message3:
        querystring['message3'] = message3
    if message10:
        querystring['message10'] = message10
    if message2:
        querystring['message2'] = message2
    if message5:
        querystring['message5'] = message5
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "discord-webhook-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def send_message(message: str, webhook_url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Allows you to send a message with a webhook. More information in the docs."
    
    """
    url = f"https://discord-webhook-api.p.rapidapi.com/send_message"
    querystring = {'message': message, 'webhook_url': webhook_url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "discord-webhook-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

