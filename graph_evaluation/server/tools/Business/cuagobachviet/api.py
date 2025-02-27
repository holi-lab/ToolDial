import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cuagobachviet(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bách Việt là đại lý cấp 1 tại Hà Nội chuyên cung cấp sản phẩm cửa gỗ công nghiệp mang thương hiệu Huge một sản phẩm của tập đoàn Austdoor Việt Nam
		Các sản phẩm cửa gỗ công nghiệp Huge Chịu nước tuyệt đối, không cong vênh, co ngót, mỗi mọt, nấm mốc, không bắt lửa. Khắc phục hoàn toàn các điểm yếu của cửa gỗ tự nhiên và cửa công nghiệp hiện tại.
		#cửa_gỗ_công_nghiệp #cửa_gỗ_bách_việt #cửa_gỗ_chịu_nước #cửa_gỗ_duratek
		Địa chỉ : 206 Phố Vọng Phương Liệt Thanh Xuân Hà Nội
		Hotline 0969412626 | 0944433433
		Email: cuagobachviet@gmail.com
		Google map https://www.google.com/maps?cid=3300295789873672605 
		Website https://cuagobachviet.vn/ 
		https://cuagobachviet.vn/bao-gia-cua-go-cong-nghiep/ 
		https://cuagobachviet.vn/cua-go-chiu-nuoc-100/ 
		https://cuagobachviet.vn/cua-go-duratek/ 
		https://sites.google.com/view/cuagobachvietvn/ 
		https://www.facebook.com/cuagobachviet 
		https://twitter.com/bachviet206 
		https://www.pinterest.com/cuagobachviet/
		https://www.youtube.com/channel/UCXxPholYmRyt3dc63MpZmmQ 
		https://tawk.to/cuagobachviet
		https://www.deviantart.com/cuagobachvietvn
		https://note.com/cuagobachviet
		https://www.mixcloud.com/cuagobachvietvn/
		https://independent.academia.edu/cuagobachviet
		https://www.reverbnation.com/artist/cuagobachviet7
		https://profile.ameba.jp/ameba/cuagobachviet
		https://fliphtml5.com/homepage/ufocg
		https://ko-fi.com/cuagobachviet#paypalModal
		https://connect.garmin.com/modern/profile/e32335ce-8478-4d16-816d-00a76aa69201
		http://wpc.hotlog.ru/profile.php?user_id=423098
		https://www.provenexpert.com/cuagobachviet/
		https://padlet.com/cuagobachviet
		https://gitlab-test.eclipse.org/cuagobachviet
		https://sketchfab.com/cuagobachviet"
    
    """
    url = f"https://cuagobachviet.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cuagobachviet.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

