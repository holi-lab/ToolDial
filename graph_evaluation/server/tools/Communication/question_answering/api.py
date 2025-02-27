import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def answer_a_question(question: str, answerlookup: bool=None, answersearch: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find an answer to a given question."
    question: The question.
        answerlookup: Whether the answer can be looked up in a given set of answers.
        answersearch: Whether the answer can be searched on the Web. If set to true, the answer might take up to a minute to show up.
        
    """
    url = f"https://webknox-question-answering.p.rapidapi.com/questions/answers"
    querystring = {'question': question, }
    if answerlookup:
        querystring['answerLookup'] = answerlookup
    if answersearch:
        querystring['answerSearch'] = answersearch
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-question-answering.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quickly_answer_a_question(question: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a short, quick answer to a simple question."
    question: A question or keywords that can be answered with simple facts.
        
    """
    url = f"https://webknox-question-answering.p.rapidapi.com/questions/quickAnswers"
    querystring = {'question': question, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-question-answering.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def suggest_conversation_topics(query: str, number: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query this endpoint to get suggests about what WebKnox can converse about."
    query: The (partial) query of the user to get suggests for.
        number: The number of suggests, must be between 1 and 25.
        
    """
    url = f"https://webknox-question-answering.p.rapidapi.com/questions/converse/suggest"
    querystring = {'query': query, }
    if number:
        querystring['number'] = number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-question-answering.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def have_a_conversation(text: str, contextid: str='qc953q672935', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using all question answering functionality to have a contextual conversation with WebKnox."
    text: The question/response by the user.
        contextid: Every conversation has a context, pass a globally unique identifier for your conversation.
        
    """
    url = f"https://webknox-question-answering.p.rapidapi.com/questions/converse"
    querystring = {'text': text, }
    if contextid:
        querystring['contextId'] = contextid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "webknox-question-answering.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

