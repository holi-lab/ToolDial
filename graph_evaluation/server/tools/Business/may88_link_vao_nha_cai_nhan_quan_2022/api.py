import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def may88_link_vao_nha_cai_2022(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "May88.live - May88 ⭐️ Link vào may88 mới nhất, web cá cược thể thao trực tuyến, Casino, Keno, Lô đề, Slot game đa dạng với nhiều thể loại cá cược.
		#May88.live #May88 #dang_ky_May88 #nhacai_May88  #Link_vào_MAY88
		Email: may88live1@gmail.com
		Hotline 0865494075
		Địa chỉ : 87 Ngô Quyền,P. Quang Trung,Hà Đông,Hà Nội
		Website https://may88.live/ 
		https://may88.live/dang-ky-may88/ 
		https://may88.live/nap-tien-may88/ 
		https://may88.live/rut-tien-may88/ 
		https://may88.live/tai-ung-dung-may88/ 
		https://twitter.com/may88live
		https://www.facebook.com/may88live/ 
		https://sites.google.com/view/may88-live/
		https://sketchfab.com/may88live
		https://www.provenexpert.com/may88-live/?mode=preview
		https://clyp.it/user/phym1swj
		https://comicvine.gamespot.com/profile/may88live/
		https://forums.giantitp.com/member.php?261635-may88live
		https://www.designspiration.com/may88live1/saves/
		https://app.lookbook.nu/may88live
		https://notionpress.com/en/ind/dashboard
		https://www.giantbomb.com/profile/may88live/
		https://challenges.openideo.com/servlet/hype/IMT?userAction=Browse&templateName=&documentId=c810fbf46c2a2d910541144fafe9bbcf
		https://www.mobypicture.com/user/may88live
		https://sitepreview-776126110.zohositescontent.com/zcms/preview
		https://www.zotero.org/may88live/cv
		https://app.glosbe.com/profile/6918672398526975206
		https://coolors.co/u/may88_live
		https://mstdn.social/web/@may88live
		https://www.webtoolhub.com/profile.aspx?user=42299681
		https://discover.events.com/profile/may88-live/3625134/savethedate/
		https://www.myvidster.com/profile/may88live
		https://gotartwork.com/Profile/may88-live/138842/
		https://www.sitejot.com/index.php
		https://roomstyler.com/users/may88live
		https://www.bitrated.com/may88live
		https://zoimas.com//profile/may88live
		https://forum.mesign.com/index.php?action=profile;u=1756679
		https://sallatunturinkoulu.purot.net/profile/may88live
		https://forum.flitetest.com/index.php?members/may88live.62837/
		http://43marks.com/may88live
		http://kwtechjobs.com/?author=272887
		http://f319.com/members/may88live.794797/"
    
    """
    url = f"https://may88-link-vao-nha-cai-nhan-quan-2022.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "may88-link-vao-nha-cai-nhan-quan-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

