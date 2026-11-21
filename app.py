# 必要なライブラリをインポート
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import altair as alt

st.set_page_config(page_title='がん罹患リスク予測', layout='wide')

# 1. 学習済みモデルと、生活習慣の補正に使う表を読み込む
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

columns = model.feature_names_in_      # 学習に使った列名
latest_year = 2023                     # 学習データの最新の年

sites = {1: '全部位', 3: '口腔・咽頭', 4: '食道', 5: '胃', 67: '大腸', 8: '肝臓', 9: '胆嚢・胆管',
         10: '膵臓', 12: '肺', 14: '乳房', 17: '子宮頸部', 18: '子宮体部', 19: '卵巣', 20: '前立腺',
         21: '膀胱', 22: '腎・尿路（膀胱除く）', 24: '甲状腺', 25: '悪性リンパ腫', 27: '白血病'}

df_rr = pd.read_csv('data/risk_factors.csv')      # 相対リスク（部位・性別・選択肢ごと）
df_prev = pd.read_csv('data/prevalence.csv')      # 各選択肢を選ぶ人の割合

# 性別によって表示しない部位
MALE_ONLY = ['前立腺']
FEMALE_ONLY = ['子宮頸部', '子宮体部', '卵巣']

# 入力の選択肢（画面に出す文字 → 表の選択肢）。先頭が一番リスクの低い選択肢
CHOICES = {
    'smoke': ['たばこ', {'吸わない': '吸わない', 'やめた': 'やめた', '吸っている': '吸っている'}],
    'alcohol': ['お酒（1合 = ビール中瓶1本程度）',
                {'飲まない・ほとんど飲まない': '飲まない', '1日1合未満': '150g未満',
                 '1日1〜2合': '150-299g', '1日2合以上': '300g以上'}],
    'salt': ['塩分の多い食品（漬物・干物・たらこ・塩辛など）',
             {'ほとんど食べない': 'ほとんど食べない', 'ときどき': 'ときどき', 'ほとんど毎日': 'ほとんど毎日'}],
    'exercise': ['体を動かす量（運動・仕事・家事を含む）', {'多い': '多い', 'ふつう': 'ふつう', '少ない': '少ない'}],
    'vegetable': ['野菜を食べる量', {'多い': '多い', 'ふつう': 'ふつう', '少ない': '少ない'}],
}
# 初期値（平均的な人に近い選択肢）
DEFAULT_INDEX = {'smoke': 0, 'alcohol': 1, 'salt': 1, 'exercise': 1, 'vegetable': 1}
FACTOR_NAMES = {'smoke': '禁煙', 'alcohol': 'お酒を控える', 'bmi': '体重を適正にする',
                'salt': '塩分を控える', 'exercise': '体を動かす', 'vegetable': '野菜を食べる'}
BEST_LEVEL = {'smoke': '吸わない', 'alcohol': '飲まない', 'bmi': '19-24.9',
              'salt': 'ほとんど食べない', 'exercise': '多い', 'vegetable': '多い'}


def bmi_level(bmi):
    if bmi < 19:
        return '19未満'
    if bmi < 25:
        return '19-24.9'
    if bmi < 30:
        return '25-29.9'
    return '30以上'


def rr(factor, level, sex, site, age):
    # 閉経後乳がんの肥満のリスクは 50歳以上にだけ使う
    if factor == 'bmi' and site == '乳房' and age < 50:
        return 1.0
    d = df_rr[(df_rr['factor'] == factor) & (df_rr['level'] == level) & (df_rr['sex'] == sex) & (df_rr['site'] == site)]
    return float(d['rr'].iloc[0]) if len(d) > 0 else 1.0


def multiplier(answers, sex, site, age):
    # 平均的な人を 1 としたときの倍率
    # 平均には喫煙者なども含まれるので、「割合 × 相対リスク」の合計（平均の相対リスク）で割る
    m = 1.0
    for factor, level in answers.items():
        p = df_prev[(df_prev['factor'] == factor) & (df_prev['sex'] == sex)]
        average_rr = 0
        for level_p, value_p in zip(p['level'], p['p']):
            average_rr = average_rr + value_p * rr(factor, level_p, sex, site, age)
        m *= rr(factor, level, sex, site, age) / average_rr
    return m


def predict_rates(sex, site, ages):
    # 年齢ごとの罹患率（人口10万対）を学習済みモデルで予測する
    rows = pd.DataFrame({
        'year': latest_year,
        'sex': sex,
        'age': [min(a // 5 * 5, 85) for a in ages],   # 学習データと同じ 5歳階級の先頭の年齢にする
        'site': site,
    })
    x = pd.get_dummies(rows, columns=['site'], dtype=int)
    x = x.reindex(columns=columns, fill_value=0)        # 学習時と同じ列にそろえる
    rates = np.exp(model.predict(x)) - 1
    rates[rates < 0] = 0
    return rates


def cumulative_risk(rates):
    # 1年ごとの罹患率を足し合わせて、その期間にかかる確率にする
    return 1 - np.exp(-np.sum(rates) / 100000)


st.title('がん罹患リスク予測')
st.caption('年齢・性別と生活習慣から、これから先にがんと診断される確率を部位ごとに予測します。')

# 計算方法の説明（ボタンを押すと開く）
with st.popover('このアプリの計算方法'):
    st.markdown('''
**1. 平均的な罹患率の予測**

全国がん登録（2016〜2023年）のデータで学習した機械学習モデルにより、
年齢・性別・部位ごとの1年あたりの罹患率（人口10万対）を予測します。

**2. 生活習慣による補正**

日本人を対象とした研究で報告されている生活習慣ごとの相対リスクを、予測した罹患率に掛けて補正します。
対象はたばこ、お酒、体格（BMI）、塩分の多い食品、体を動かす量、野菜を食べる量です。
平均的な罹患率には各習慣の人が一定の割合で含まれているため、選択肢ごとの割合で平均が 1 倍になるよう調整したうえで掛けています。

**3. 期間内の確率の算出**

1年ごとの罹患率を合計し、指定した期間（10年または20年）にがんと診断される確率に換算します。
''')

# 2. サイドバーで入力を受け取る
st.sidebar.header('あなたについて')
age = st.sidebar.slider('年齢', min_value=20, max_value=84, value=50)
sex_name = st.sidebar.radio('性別', ['男性', '女性'], horizontal=True)
sex = 1 if sex_name == '男性' else 0
period = st.sidebar.radio('予測する期間', ['今後10年', '今後20年'], horizontal=True)
years = 10 if period == '今後10年' else 20

height = st.sidebar.number_input('身長（cm）', min_value=120.0, max_value=220.0, value=170.0 if sex == 1 else 158.0, step=0.5)
weight = st.sidebar.number_input('体重（kg）', min_value=30.0, max_value=200.0, value=67.0 if sex == 1 else 53.0, step=0.5)
bmi = weight / (height / 100) ** 2
st.sidebar.caption(f'BMI：{bmi:.1f}')

st.sidebar.divider()
st.sidebar.header('生活習慣')
answers = {}
for factor in CHOICES:
    label = CHOICES[factor][0]
    options = CHOICES[factor][1]
    choice = st.sidebar.radio(label, list(options.keys()), index=DEFAULT_INDEX[factor])
    answers[factor] = options[choice]
answers['bmi'] = bmi_level(bmi)

ages = list(range(age, age + years))

# 3. 部位ごとに予測（学習済みモデル → 生活習慣で補正）
results = []
for code, name in sites.items():
    if (sex == 0 and name in MALE_ONLY) or (sex == 1 and name in FEMALE_ONLY):
        continue
    rates = predict_rates(sex, code, ages)
    m = multiplier(answers, sex, name, age)
    results.append([name, cumulative_risk(rates) * 100, cumulative_risk(rates * m) * 100, m])

df_result = pd.DataFrame(results, columns=['部位', '平均的な人（%）', 'あなた（%）', '平均との比'])
total = df_result[df_result['部位'] == '全部位'].iloc[0]
df_sites = df_result[df_result['部位'] != '全部位'].sort_values('あなた（%）', ascending=False)

# リスクの目安（平均との比で色分け）
ratio = total['平均との比']
if ratio >= 1.5:
    level_text = ':red-background[平均よりかなり高め]'
elif ratio >= 1.1:
    level_text = ':orange-background[平均よりやや高め]'
elif ratio > 0.9:
    level_text = ':blue-background[ほぼ平均]'
else:
    level_text = ':green-background[平均より低め]'

# 4. 結果のサマリー
with st.container(border=True):
    st.subheader(f'{age}歳・{sex_name}の予測結果　{level_text}')
    col1, col2, col3 = st.columns(3)
    col1.metric(f'今後{years}年でいずれかのがんと診断される確率', f'{total["あなた（%）"]:.1f} %',
                f'{total["あなた（%）"] - total["平均的な人（%）"]:+.1f} pt（平均との差）', delta_color='inverse', border=True)
    col2.metric('約何人に1人か', f'{100 / total["あなた（%）"]:.0f} 人に1人', border=True)
    col3.metric('同じ年齢・性別の平均と比べて', f'{ratio:.2f} 倍', border=True)

    top3 = df_sites.head(3)['部位'].tolist()
    st.write(f'確率が高い部位は **{"・".join(top3)}** です。')

tab1, tab2, tab3 = st.tabs(['部位別の確率', '習慣を変えると', '年齢による変化'])

# 5. 部位別の確率
with tab1:
    high = df_sites[df_sites['平均との比'] >= 1.2].sort_values('平均との比', ascending=False)
    if len(high) > 0:
        text = []
        for name, r in zip(high['部位'], high['平均との比']):
            text.append(f'{name}（{r:.1f}倍）')
        st.warning('平均より高めの部位：' + '、'.join(text))

    # 棒グラフ用に「部位・区分・確率」の形にする
    df_chart = df_sites.melt(id_vars=['部位'], value_vars=['あなた（%）', '平均的な人（%）'],
                             var_name='区分', value_name='確率')
    df_chart['区分'] = df_chart['区分'].map({'あなた（%）': 'あなた', '平均的な人（%）': '平均的な人'})
    df_chart['ラベル'] = df_chart['確率'].map('{:.2f}%'.format)

    order = df_sites['部位'].tolist()     # 「あなた」の確率が高い順
    bars = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X('確率:Q', title='確率（%）'),
        y=alt.Y('部位:N', sort=order, title=None),
        yOffset=alt.YOffset('区分:N', sort=['あなた', '平均的な人']),
        color=alt.Color('区分:N', scale=alt.Scale(domain=['あなた', '平均的な人'], range=['#D9485F', '#C8CDD3']),
                        legend=alt.Legend(title=None, orient='top')),
    )
    labels = bars.mark_text(align='left', dx=4, fontSize=12).encode(text='ラベル:N', color=alt.value('#2B2B2B'))
    st.altair_chart((bars + labels).properties(height=len(order) * 42), width='stretch')

# 6. 習慣を変えたらどうなるか
with tab2:
    improve = []
    for factor, level in answers.items():
        if level == BEST_LEVEL[factor]:
            continue
        new_answers = dict(answers)
        new_answers[factor] = BEST_LEVEL[factor]
        rates = predict_rates(sex, 1, ages)
        new_risk = cumulative_risk(rates * multiplier(new_answers, sex, '全部位', age)) * 100
        improve.append([FACTOR_NAMES[factor], new_risk, new_risk - total['あなた（%）']])

    if improve:
        df_improve = pd.DataFrame(improve, columns=['取り組み', '変えた後の確率（%）', '差（pt）']).sort_values('差（pt）')
        st.write(f'1つだけ変えた場合の、今後{years}年でいずれかのがんと診断される確率です。')
        cols = st.columns(3)
        for i in range(len(df_improve)):
            row = df_improve.iloc[i]
            cols[i % 3].metric(row['取り組み'], f'{row["変えた後の確率（%）"]:.1f} %', f'{row["差（pt）"]:.1f} pt',
                               delta_color='inverse', border=True)
    else:
        st.success('入力した項目は、すべてリスクが低い選択肢です。この習慣を続けましょう。')

# 7. 年齢による変化
with tab3:
    site_name = st.selectbox('部位を選ぶ', df_sites['部位'].tolist())
    for code, name in sites.items():
        if name == site_name:
            site_code = code
    curve_ages = list(range(20, 90, 5))
    curve = pd.DataFrame({'年齢': curve_ages})
    colors = []
    for s, label, color in [(1, '男性', '#3B7DD8'), (0, '女性', '#D9485F')]:
        if (s == 0 and site_name in MALE_ONLY) or (s == 1 and site_name in FEMALE_ONLY):
            continue
        curve[label] = predict_rates(s, site_code, curve_ages)
        colors.append(color)
    st.line_chart(curve.set_index('年齢'), y_label='罹患率（人口10万対）', color=colors)
    st.caption(f'平均的な人について、学習済みモデルで {latest_year}年の水準を予測した1年あたりの罹患率です。')

st.info('医学的な診断ではありません。家族歴・感染（ピロリ菌、肝炎ウイルスなど）・持病などは反映していません。'
        '「全部位」は同じ人が複数のがんにかかった場合も数えているため、目安として見てください。'
        '気になることがある場合は医療機関やがん検診を利用してください。')

st.caption('出典')
st.caption('・罹患率：国立がん研究センターがん情報サービス「がん統計」（全国がん登録）  \n'
           '　https://ganjoho.jp/reg_stat/statistics/data/dl/index.html  \n'
           '・相対リスク：国立がん研究センター 多目的コホート研究（JPHC Study）ほか  \n'
           '・喫煙率・飲酒・肥満の割合：厚生労働省「国民健康・栄養調査」')
