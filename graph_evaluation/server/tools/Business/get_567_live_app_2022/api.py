import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_567_live_t_i_app_m_i_nh_t_2022_t_i_trang_ch_ch_nh_th_c(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "567 Live có trang chủ 567live.io duy nhất chính xác. Cung cấp đủ link tải 567 live app an toàn cho bạn phiên bản mới nhất.
		Địa chỉ : 99 Giải Phóng, Đồng Tâm, Hai Bà Trưng, Hà Nội
		Google map https://www.google.com/maps?cid=7118718670140148249 
		Hotline 0978720550
		Gmail:  567liveapp@gmail.com
		Website https://567live.io/ 
		https://sites.google.com/view/567live-app/
		https://567live.io/tai-ve/567live-apk 
		https://567live.io/tai-ve/567live-ios
		https://567live.io/thu-muc/app-live/ 
		https://folkd.com/user/567liveapp
		https://sketchfab.com/567liveapp
		https://forums.iis.net/members/567liveapp.aspx
		https://wefunder.com/567live
		https://www.ultimate-guitar.com/u/567liveapp
		https://www.thingiverse.com/567liveapp/designs
		https://www.tickaroo.com/user/61fe0be0ed68b63b06bd40d5
		https://www.goodreads.com/user/show/147284265-567live
		https://gifyu.com/567liveapp
		https://ignitiondeck.com/id/dashboard/?backer_profile=153894
		https://www.sqlservercentral.com/forums/user/567liveapp
		https://abroadsanjal.com/index.php?qa=user&qa_1=567live
		https://beermapping.com/account/567liveapp
		https://www.zoimas.com/profile/567liveapp1/about
		https://www.eduvision.edu.pk/counseling/index.php?qa=user&qa_1=567liveapp
		https://567live.hpage.com/
		https://www.leetchi.com/en/c/5Yj6J40r
		https://public.tableau.com/app/profile/567live
		https://www.brownbook.net/user-profile/4782891
		https://amara.org/en/profiles/profile/MndaPixiSx4HXYOyjdeCtoryc1vYtjp933bdJZs6MYo/
		https://cycling74.com/author/6201cfb5d540c65659fc25e7
		https://api.rakuten.net/user/567liveapp
		https://lnk.bio/567live
		https://www.pozible.com/profile/567live
		https://new.edmodo.com/home
		http://www.nfomedia.com/profile?uid=rKfQfiD&result=ep5dbp81
		https://degreed.com/profile/567liveapp/overview
		https://bittube.tv/profile/567live
		https://git.feneas.org/567liveapp
		https://forum.singaporeexpats.com/memberlist.php?mode=viewprofile&u=370389
		https://sallatunturinkoulu.purot.net/profile/567liveapp
		https://liveviewsports.com/community/profile/567live/
		http://q2a.sydt.com.tw/index.php?qa=user&qa_1=567live"
    
    """
    url = f"https://567-live-app-2022.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "567-live-app-2022.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

