import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def parser_2_0(lang: str, accept: str='application/json', txt: str="Robert Downey Jr has topped Forbes magazine's annual list of the highest paid actors for the second year in a row.", ud: str=None, verbose: str='y', txtf: str='plain', uw: str='n', tt: str=None, st: str='n', rt: str='n', dm: str='s', timeref: str=None, egp: str='n', sdg: str='l', sm: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lemmatization, PoS and Parsing is the name of MeaningCloud' API for the different basic linguistic modules."
    lang: It specifies the language in which the text must be analyzed.
        txt: Input text that's going to be analyzed
        ud: The user dictionary allows to include user-defined entities and concepts in the analysis.
        verbose: Verbose mode. When active, it shows additional information about the morphosyntactic tagsets.
        txtf: It specifies if the text included in the txt parameter uses markup language that needs to be interpreted.
        uw: Deal with unknown words.
        tt: The list of topic types to extract will be specified through a string with the letters assigned to each one of the topic types that are to be extracted.
        st: Show subtopics. This parameter will indicate if subtopics are to be shown.
        rt: Deal with relaxed typography.
        dm: Type of disambiguation applied. It is accumulative, that is, the semantic disambiguation mode will also include morphosyntactic disambiguation.
        timeref: This value allows to set a specific time reference to detect the actual value of all the relative time expressions detected in the text. It only applies when time expressions are enabled in tt. Format: YYYY-MM-DD hh:mm:ss GMT±HH:MM
        egp: Expand global polarity. This mode allows you to choose between two different algorithms for the polarity detection of entities and concepts.
        sdg: Semantic disambiguation grouping. This parameter will only apply when semantic disambiguation is activated (dm=s).
        sm: Sentiment model chosen. If sent empty, sentiment analysis info will not be included in the response.
        
    """
    url = f"https://lemmatization-pos-parsing.p.rapidapi.com/parser-2.0"
    querystring = {'lang': lang, }
    if accept:
        querystring['accept'] = accept
    if txt:
        querystring['txt'] = txt
    if ud:
        querystring['ud'] = ud
    if verbose:
        querystring['verbose'] = verbose
    if txtf:
        querystring['txtf'] = txtf
    if uw:
        querystring['uw'] = uw
    if tt:
        querystring['tt'] = tt
    if st:
        querystring['st'] = st
    if rt:
        querystring['rt'] = rt
    if dm:
        querystring['dm'] = dm
    if timeref:
        querystring['timeref'] = timeref
    if egp:
        querystring['egp'] = egp
    if sdg:
        querystring['sdg'] = sdg
    if sm:
        querystring['sm'] = sm
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "lemmatization-pos-parsing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

