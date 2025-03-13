# Spotify
git add . 
git commit -m "Update" 
git push origin main


# install
pip install scipy
pip install pandas
pip install seaborn
pip install scikit-learn

# Explanation of the features

⭕id - ID，唯一標識符 / ID, 고유 식별자
⭕year - 發行年份 / 발매 연도
❌name - 名稱（歌曲名稱） / 이름 (노래 제목)
❌artists - 藝術家（歌手/樂隊） / 아티스트 (가수/밴드)
❌release_date - 發行日期 / 발매일
✔️valence - 愉悅度（衡量歌曲情緒的指標，數值越高越積極） / 밝기 (노래의 분위기를 측정하는 지표, 값이 높을수록 긍정적임)
✔️acousticness - 原聲度（衡量歌曲是否為原聲音樂，值越高表示越原聲） / 어쿠스틱함 (노래가 어쿠스틱한지 측정하는 값, 높을수록 어쿠스틱 성향이 강함)
✔️danceability - 舞蹈性（衡量歌曲是否適合跳舞） / 댄서빌리티 (노래가 춤추기에 적합한지 측정하는 값)
✔️duration_ms - 時長（以毫秒為單位） / 재생 시간 (밀리초 단위)
✔️energy - 能量值（衡量歌曲的活力，數值越高越有力量） / 에너지 (노래의 활력을 측정하는 값, 높을수록 강렬함)
✔️explicit - 是否含有不當內容（0=無，1=有） / 비속어 포함 여부 (0=없음, 1=있음)
✔️instrumentalness - 器樂性（衡量歌曲是否為純音樂） / 기악성 (노래가 연주곡인지 측정하는 값, 높을수록 가사가 적음)
✔️key - 調（音樂調性，如C大調、D小調） / 키 (음악의 조성, 예: C장조, D단조)
✔️liveness - 現場感（是否為現場錄製，數值越高越像現場演出） / 라이브 감성 (라이브 녹음 여부, 값이 높을수록 현장감 있음)
✔️loudness - 音量（分貝單位，衡量歌曲的平均響度） / 음량 (데시벨 단위, 평균 볼륨 수준)
✔️mode - 調式（0=小調，1=大調） / 모드 (0=단조, 1=장조)
✔️speechiness - 語音性（衡量歌曲中的語音成分，如說唱） / 스피치니스 (노래에 말하는 요소가 포함된 정도, 예: 랩)
✔️tempo - 節奏（BPM，每分鐘節拍數） / 템포 (BPM, 분당 비트 수)
✅popularity - 流行度（衡量歌曲的受歡迎程度） / 인기도 (노래의 인기 정도를 나타내는 값)