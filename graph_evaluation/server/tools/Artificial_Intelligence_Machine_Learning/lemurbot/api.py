import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_category_group(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific category group by it's id and view it's details.
		You are able to view your own category groups and locked category groups.
		'Locked' category groups are a master set of records available to link to your bots"
    id: The id of CategoryGroup
        
    """
    url = f"https://lemurbot.p.rapidapi.com/category-groups/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_category(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific category by it's id and view it's details"
    id: The id of Category
        
    """
    url = f"https://lemurbot.p.rapidapi.com/categories/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_bot_property(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific bot property by it's id and view it's details"
    id: The id of BotProperty
        
    """
    url = f"https://lemurbot.p.rapidapi.com/bot-properties/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_bot_properties(limit: int=20, value: str=None, name: str='name', page: int=1, is_id: int=None, order: str='id', bot_id: int=None, dir: str='asc', bot: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of bot properties that belong to bots created by the user (not locked bots)."
    limit: the number of records to return
        value: Value of the bot propery - this will perform a partial search
        name: Name of the bot propery
        page: the page number of results to start from
        is_id: The id of the bot property
        order: the search field you wish to order by - the available field names are the sames ones you can search by
        bot_id: bot_id of the bot property
        dir: the direction of the ordering
        bot: Bot slug of the bot
        
    """
    url = f"https://lemurbot.p.rapidapi.com/bot-properties"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if value:
        querystring['value'] = value
    if name:
        querystring['name'] = name
    if page:
        querystring['page'] = page
    if is_id:
        querystring['id'] = is_id
    if order:
        querystring['order'] = order
    if bot_id:
        querystring['bot_id'] = bot_id
    if dir:
        querystring['dir'] = dir
    if bot:
        querystring['bot'] = bot
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_bots(is_id: int=None, dir: str='asc', slug: str=None, order: str='id', locked: bool=None, title: str=None, page: int=1, description: str=None, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of bots that are available to this user.
		These will either be bots created by the user or 'locked' public bots which can be chatted to but not editted. (e.g. 1)"
    is_id: The id of Bot
        dir: the direction of the ordering
        slug: Slug of the bot (e.g. dilly)
        order: the search field you wish to order by - the available field names are the sames ones you can search by
        locked: search for bots which are locked or bots which are not locked (e.g. false)
        title: Title of the bot (e.g. Dilly)
        page: the page number of results to start from
        description: Perform a partial search on the description of the bot (e.g. demo bot)
        limit: the number of records to return
        
    """
    url = f"https://lemurbot.p.rapidapi.com/bots"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    if dir:
        querystring['dir'] = dir
    if slug:
        querystring['slug'] = slug
    if order:
        querystring['order'] = order
    if locked:
        querystring['locked'] = locked
    if title:
        querystring['title'] = title
    if page:
        querystring['page'] = page
    if description:
        querystring['description'] = description
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_bot(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific bot by its id. (e.g. 1)"
    id: The id of Bot
        
    """
    url = f"https://lemurbot.p.rapidapi.com/bots/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_category_groups(locked: bool=None, order: str='id', name: str='hello', page: int=1, is_id: int=None, dir: str='asc', limit: int=20, slug: str='hello', description: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Categories are organised into category groups. For example if you have a set of categories to handle questions on the weather these could be grouped into a category group called 'weather-categories'.
		You are able to view your own category groups and locked category groups.
		'Locked' category groups are a master set of records available to link to your bots"
    locked: Locked category groups can be linked to bot so that you can access their knowledge categories but you cannot cannot update, delete or access this group or it's categories
        order: the search field you wish to order by - the available field names are the sames ones you can search by
        name: the name of the category group - this search is a partial search
        page: the page number of results to start from
        is_id: The id of the category group
        dir: the direction of the ordering
        limit: the number of records to return
        slug: the slug of the category group - this search is an exact search
        description: the description of the category group - this search is a partial search
        
    """
    url = f"https://lemurbot.p.rapidapi.com/category-groups"
    querystring = {}
    if locked:
        querystring['locked'] = locked
    if order:
        querystring['order'] = order
    if name:
        querystring['name'] = name
    if page:
        querystring['page'] = page
    if is_id:
        querystring['id'] = is_id
    if dir:
        querystring['dir'] = dir
    if limit:
        querystring['limit'] = limit
    if slug:
        querystring['slug'] = slug
    if description:
        querystring['description'] = description
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_clients(order: str='id', slug: str=None, is_id: int=None, limit: int=20, dir: str='asc', page: int=1, is_banned: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Clients talk to bots. This return a list of all clients that have spoken your bots. Clients can talk to your bots or the locked bots.
		'Locked' bots are a set of public bots available to link to your clients in addition to bots you create."
    order: the search field you wish to order by - the available field names are the sames ones you can search by
        slug: The slug of the client
        is_id: The id of the client
        limit: the number of records to return
        dir: the direction of the ordering
        page: the page number of results to start from
        is_banned: The banned status of this client
        
    """
    url = f"https://lemurbot.p.rapidapi.com/clients"
    querystring = {}
    if order:
        querystring['order'] = order
    if slug:
        querystring['slug'] = slug
    if is_id:
        querystring['id'] = is_id
    if limit:
        querystring['limit'] = limit
    if dir:
        querystring['dir'] = dir
    if page:
        querystring['page'] = page
    if is_banned:
        querystring['is_banned'] = is_banned
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_bot_category_group(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific bot category group by it's id and view it's details"
    id: The id of BotCategoryGroup
        
    """
    url = f"https://lemurbot.p.rapidapi.com/bot-category-groups/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_client(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific client by it's id and view it's details"
    id: The id of Client
        
    """
    url = f"https://lemurbot.p.rapidapi.com/clients/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_bot_category_groups(category_group: str=None, dir: str='asc', limit: int=20, order: str='id', is_id: int=None, bot: str=None, category_group_id: int=None, bot_id: int=None, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of category groups which are linked to user owned bots (not locked bots)."
    category_group: Category group slug of the category group in the bot category group
        dir: the direction of the ordering
        limit: the number of records to return
        order: the search field you wish to order by - the available field names are the sames ones you can search by
        is_id: The id of the bot category group
        bot: Bot slug of the bot in the bot category group
        category_group_id: category_group_id of the category group
        bot_id: bot_id of the bot category group
        page: the page number of results to start from
        
    """
    url = f"https://lemurbot.p.rapidapi.com/bot-category-groups"
    querystring = {}
    if category_group:
        querystring['category_group'] = category_group
    if dir:
        querystring['dir'] = dir
    if limit:
        querystring['limit'] = limit
    if order:
        querystring['order'] = order
    if is_id:
        querystring['id'] = is_id
    if bot:
        querystring['bot'] = bot
    if category_group_id:
        querystring['category_group_id'] = category_group_id
    if bot_id:
        querystring['bot_id'] = bot_id
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_categories(limit: int=20, category_group_id: int=None, dir: str='asc', topic: str=None, page: int=1, order: str='id', that: str=None, template: str=None, pattern: str='test', is_id: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Categories are pieces of knowledge used by the bots to respond to the user.
		Create and update categories inorder to teach and train your bot."
    limit: the number of records to return
        category_group_id: The id of the category group
        dir: the direction of the ordering
        topic: The active topic of the conversation - this search is a full term search
        page: the page number of results to start from
        order: the search field you wish to order by - the available field names are the sames ones you can search by
        that: The that is the previous bot output/response to match on - this search is a partial search
        template: The template is the raw output that is parsed and used to generate a response - this search is a partial search
        pattern: The pattern is the user input to match on - this search is a partial search
        is_id: The id of the category
        
    """
    url = f"https://lemurbot.p.rapidapi.com/categories"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if category_group_id:
        querystring['category_group_id'] = category_group_id
    if dir:
        querystring['dir'] = dir
    if topic:
        querystring['topic'] = topic
    if page:
        querystring['page'] = page
    if order:
        querystring['order'] = order
    if that:
        querystring['that'] = that
    if template:
        querystring['template'] = template
    if pattern:
        querystring['pattern'] = pattern
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_conversation(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific conversation by it's id and view it's details"
    id: The id of Conversation
        
    """
    url = f"https://lemurbot.p.rapidapi.com/conversations/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_conversations(page: int=1, limit: int=20, order: str='id', bot: str=None, client: str=None, bot_id: int=None, client_id: int=None, slug: str=None, is_id: int=None, dir: str='asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Conversations are created between clients and bots.
		Users can only view conversations between their clients and bots."
    page: the page number of results to start from
        limit: the number of records to return
        order: the search field you wish to order by - the available field names are the sames ones you can search by
        bot: The bot slug of the conversation - conversation are linked to bots
        client: The client slug of the conversation - conversation are linked to clients
        bot_id: The bot_id of the conversation - conversation are linked to bots
        client_id: The client_id of the conversation - conversation are linked to clients
        slug: The slug of the conversation
        is_id: The id of the conversation
        dir: the direction of the ordering
        
    """
    url = f"https://lemurbot.p.rapidapi.com/conversations"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    if order:
        querystring['order'] = order
    if bot:
        querystring['bot'] = bot
    if client:
        querystring['client'] = client
    if bot_id:
        querystring['bot_id'] = bot_id
    if client_id:
        querystring['client_id'] = client_id
    if slug:
        querystring['slug'] = slug
    if is_id:
        querystring['id'] = is_id
    if dir:
        querystring['dir'] = dir
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_conversation_turns(limit: int=20, conversation: str=None, conversation_id: int=None, dir: str='asc', input: str=None, page: int=1, output: str=None, is_id: int=None, order: str='id', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Each conversation has conversation turns.
		Each turn represents an input from the client and the output from the bot."
    limit: the number of records to return
        conversation: The slug of the conversation - conversation are linked to conversation turns
        conversation_id: The id of the conversation - conversation are linked to conversation turns
        dir: the direction of the ordering
        input: What was said to the bot by the client - this is a partial search
        page: the page number of results to start from
        output: What was said by the bot in response to the client input - this is a partial search
        is_id: The id of the conversation turn
        order: the search field you wish to order by - the available field names are the sames ones you can search by
        
    """
    url = f"https://lemurbot.p.rapidapi.com/conversation-turns"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if conversation:
        querystring['conversation'] = conversation
    if conversation_id:
        querystring['conversation_id'] = conversation_id
    if dir:
        querystring['dir'] = dir
    if input:
        querystring['input'] = input
    if page:
        querystring['page'] = page
    if output:
        querystring['output'] = output
    if is_id:
        querystring['id'] = is_id
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_a_conversation_turn(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a specific conversation turn by it's id and view it's details"
    id: The id of ConversationTurn
        
    """
    url = f"https://lemurbot.p.rapidapi.com/conversation-turns/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_chat_detail(client: str, bot: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the details for an active chat session.
		Send a bot slug and client slug as query parameters and the request will return the details from the last active chat session."
    client: The slug of the client
        bot: The slug of the bot
        
    """
    url = f"https://lemurbot.p.rapidapi.com/chat/detail"
    querystring = {'client': client, 'bot': bot, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_chat_log(client: str, bot: str, dir: str='asc', page: int=1, limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the logs for an active chat session.
		Send a bot slug and client slug as query parameters and the request will return the logs from the last active chat session."
    client: The slug of the client
        bot: The slug of the bot
        dir: the direction of the ordering
        page: the page number of results to start from
        limit: the number of records to return
        
    """
    url = f"https://lemurbot.p.rapidapi.com/chat/log"
    querystring = {'client': client, 'bot': bot, }
    if dir:
        querystring['dir'] = dir
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemurbot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

