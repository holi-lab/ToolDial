import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sentiment_2_1(lang: str, txt: str, model: str, accept: str='application/json', dm: str='s', sdg: str='l', txtf: str='plain', cont: str=None, of: str='json', uw: str='n', rt: str='n', egp: str='n', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sentiment Analysis 2.1"
    lang: Language in which to analyze the text
        txt: Text to analyze
        model: Sentiment model to use in the analysis
        dm: Type of disambiguation applied. It is accumulative, that is, the semantic disambiguation mode will also include morphosyntactic disambiguation. Possible values: n (no disambiguation), m (morphosyntactic disambiguation), s (semantic disambiguation)
        sdg: Semantic disambiguation grouping. This parameter will only apply when semantic disambiguation is activated (dm=s). Possible values: n (none), g (global intersection), t (intersection by type), l (intersection by type - smallest location)
        txtf: Text format
        cont: Disambiguation context. Context prioritization for entity semantic disambiguation
        of: Output format
        uw: Deal with unknown words. This feature adds a stage to the topic extraction in which the engine, much like a spellchecker, tries to find a suitable analysis to the unknown words resulted from the initial analysis assignment. It is specially useful to decrease the impact typos have in text analyses. Possible values: y or n
        rt: This parameter indicates how reliable the text to analyze is (as far as spelling, typography, etc. are concerned), and influences how strict the engine will be when it comes to take these factors into account in the analysis. Possible values: y, n or u (enabled just for user dictionary)
        egp: Expand global polarity. This mode allows you to choose between two different algorithms for the polarity detection of entities and concepts. Enabling the parameter gives less weight to the syntactic relationships, so it's recommended for short texts with unreliable typography. Possible values: y or n
        
    """
    url = f"https://sentiment-analysis.p.rapidapi.com/sentiment-2.1"
    querystring = {'lang': lang, 'txt': txt, 'model': model, }
    if accept:
        querystring['Accept'] = accept
    if dm:
        querystring['dm'] = dm
    if sdg:
        querystring['sdg'] = sdg
    if txtf:
        querystring['txtf'] = txtf
    if cont:
        querystring['cont'] = cont
    if of:
        querystring['of'] = of
    if uw:
        querystring['uw'] = uw
    if rt:
        querystring['rt'] = rt
    if egp:
        querystring['egp'] = egp
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sentiment-analysis.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

