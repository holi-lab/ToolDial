import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_labeled_data(q: str='{"object_analysis": {"$elemMatch": {"$elemMatch": {"classname": "cat"}}}, "width": {"$gt": 100}, "label_quality_estimation": {"$gt": 0.5}}', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns 25 labeled data samples (or less if the page does not have 25 labeled data samples) and the total length of the search query result."
    q: [MongoDB Query Language query](https://www.mongodb.com/docs/manual/tutorial/query-documents/).

Available attributes for filtering: 'object_analysis', 'width', 'height', 'label_quality_estimation'.

Options for 'classname' within 'object_analysis': 'airplane', 'apple', 'backpack', 'banana', 'baseballbat', 'baseballglove', 'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cellphone', 'chair', 'clock', 'couch', 'cow', 'cup', 'diningtable', 'dog', 'donut', 'elephant', 'firehydrant', 'fork', 'frisbee', 'giraffe', 'hairdrier', 'handbag', 'horse', 'hotdog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven', 'parkingmeter', 'person', 'pizza', 'pottedplant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard', 'skis', 'snowboard', 'spoon', 'sportsball', 'stopsign', 'suitcase', 'surfboard', 'teddybear', 'tennisracket', 'tie', 'toaster', 'toilet', 'toothbrush', 'trafficlight', 'train', 'truck', 'tv', 'umbrella', 'vase', 'wineglass', 'zebra'.

Label quality estimation value range is [0, 1]. 0 means bad label quality and 1 means good label quality.
        
    """
    url = f"https://cortex4.p.rapidapi.com/get-labeled-data"
    querystring = {}
    if q:
        querystring['q'] = q
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cortex4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

