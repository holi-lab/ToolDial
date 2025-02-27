import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def zeno(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "<?php
		
		$request = new HttpRequest();
		$request->setUrl('https://zeno-media1.p.rapidapi.com/');
		$request->setMethod(HTTP_METH_GET);
		
		$request->setHeaders([
			'Authorization' => '<REQUIRED>',
			'X-RapidAPI-Key' => 'f7de183c6dmsh5b1dbec0a9bd773p1c5a2ajsn58e8c3333ad3',
			'X-RapidAPI-Host' => 'zeno-media1.p.rapidapi.com'
		]);
		
		try {
			$response = $request->send();
		
			echo $response->getBody();
		} catch (HttpException $ex) {
			echo $ex;
		}"
    
    """
    url = f"https://4realrecords.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "4realrecords.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

