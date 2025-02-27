import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def realise(subject: str, verb: str, object: str, subjdet: str=None, subjnum: str=None, objdet: str=None, objnum: str=None, progressive: str=None, objmod: str=None, sentencetype: str=None, negated: str=None, tense: str=None, passive: str=None, modal: str=None, perfect: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API can realise simple sentences given a subject, a verb and an object. Optional additional parameters allow to specify features like tense, number, voice, sentence type."
    subject: any noun, any pronoun
        verb: any verb, including phrasal verbs like „start out“
        object: any noun or pronoun. If the object is a pronoun set objdet=- and use the base form of the pronoun (e.g. she instead of her). Possible base forms are: I, you, he, she, it, we, they. This will be automatically replaced by me, you, him, her, it, us, them. Set objnum=plural to get the plural forms (e. g. object=he and objdet=- and objnum=plural yields them).
        subjdet: Specifies the subject’s number (a, the, –). Default is „the“. Use subjdet=- to generate a subject without determiner.
        subjnum: Specifies the subject’s number (singular, plural). A value of plural is overridden by subjdet=a.
        objdet: Specifies the object’s determiner (a, the, –). Default is „a“. Use objdet=- to generate an object without determiner.
        objnum: Specifies the object’s number (singular, plural). Default is singular. A value of plural is overridden by objdet=a.
        progressive: &progressive=progressive sets the progressive (continuous) tense
        objmod: Specifies an adjective modifying the object: any adjective
        sentencetype: Specifies the sentence type (yesno, whatobj, whosubj). Default is a declarative sentence. &sentencetype=yesno generates a yes/no-question, whatobj generates a WH-question for the object, whosubj generates a WH-question for the subject
        negated: &negated=negated generates a negated sentence
        tense: Specifies the verb’s tense (present, past, future). Default is „present“
        passive: &passive=passive generates a sentence in passive voice. The object is set as subject and the subject becomes the by-object.
        modal: Specifies a modal verb modifying the verb (can, may, must, ought, shall, should, would). Only allowed for present tense. If tense=past or tense=future is set then the parameter modal will be ignored.
        perfect: &perfect=perfect sets the perfect tense
        
    """
    url = f"https://linguatools-sentence-generating.p.rapidapi.com/realise"
    querystring = {'subject': subject, 'verb': verb, 'object': object, }
    if subjdet:
        querystring['subjdet'] = subjdet
    if subjnum:
        querystring['subjnum'] = subjnum
    if objdet:
        querystring['objdet'] = objdet
    if objnum:
        querystring['objnum'] = objnum
    if progressive:
        querystring['progressive'] = progressive
    if objmod:
        querystring['objmod'] = objmod
    if sentencetype:
        querystring['sentencetype'] = sentencetype
    if negated:
        querystring['negated'] = negated
    if tense:
        querystring['tense'] = tense
    if passive:
        querystring['passive'] = passive
    if modal:
        querystring['modal'] = modal
    if perfect:
        querystring['perfect'] = perfect
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linguatools-sentence-generating.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

