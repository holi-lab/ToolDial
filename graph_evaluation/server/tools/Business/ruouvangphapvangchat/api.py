import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ruouvangphapvangchat(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ruouvangphapvangchat  - Rượu vang Pháp chuyên cung cấp nhập khẩu chính hãng với các loại rượu vang nổi tiếng như: Château Lafite-Rothschild, Château Latour, Château Mouton-Rothschild và Chateau Vignot , chế độ bảo quản đúng tiêu chuẩn. Giá tốt nhất thị trường. Hệ thống phân phối toàn quốc Tphcm, Hà Nội, Đà Nẵng, Đồng Nai, Bắc Ninh..... Giao hàng miễn phí. #ruouvangphapvangchat  #Rượu_vang_pháp #vangchatcomvn #giaruouvangphap 
		
		Địa chỉ 65 Ngách 323/83 Đường Xuân Đỉnh, Xuân Đỉnh, Từ Liêm, Hà Nội
		Hotline 84849788111
		Gmail: Royalwines.hn@gmail.com
		Google map https://www.google.com/maps?cid=32658451913975955   
		Website https://vangchat.com.vn/quoc-gia/ruou-vang-phap/ 
		https://vi.wikipedia.org/wiki/Rượu_vang_Pháp  
		https://www.beatstars.com/ruouvangphapvangchat
		https://hearthis.at/ruouvangphapvangchat/set/ruouvangphapvangchat/
		https://www.question2answer.org/qa/user/ruouvangphapvangchat
		https://muckrack.com/ruouvangphap-vangchat/bio
		https://reedsy.com/discovery/user/ruouvangphapvangchat
		https://beacons.ai/ruouvangphapvangchat
		https://roundme.com/@ruouvangphapvangchat/about
		https://leetcode.com/ruouvangphapvangchat/
		https://cplusplus.com/user/ruouvangphapvangchat/
		https://community.dynamics.com/members/ruouvangphapvangchat
		https://themehunt.com/profile/ruouvangphapvangchat
		https://www.picfair.com/users/ruouvangphapvangchat
		https://www.furaffinity.net/user/ruouvangphapvangchat/
		https://folkd.com/user/ruouvangphapvangchat
		https://xipnios.tribe.so/user/ruouvangphapvangchat
		https://www.intensedebate.com/profiles/ruouvangphapvangchat
		https://www.lifeofpix.com/photographers/ruouvangphapvangchat/
		https://www.emoneyspace.com/ruouvangphapvangchat
		https://tapas.io/ruouvangphapvangchat"
    
    """
    url = f"https://ruouvangphapvangchat.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ruouvangphapvangchat.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

