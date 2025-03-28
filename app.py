import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'

DATA = 'welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'


@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, 'r', encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

with st.sidebar:
    st.header('Powiedz nam coś o sobie')
    st.markdown('Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania')
    age = st.selectbox('Wiek', ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox('Wykształcenie', ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox('Ulubione zwierzęta', ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox('Ulubione miejsce', ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio('Płeć', ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()


predicted_cluster_id = predict_model(model, data=person_df)['Cluster'].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f'Najbliżej Ci do grupy {predicted_cluster_data["name"]}')
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df['Cluster'] == predicted_cluster_id]
st.metric('Liczba twoich znajomych', len(same_cluster_df))

st.header('Porównanie z innymi grupami')
other_clusters = all_df['Cluster'].unique()
cluster_sizes = all_df.groupby('Cluster').size()

# Tworzenie słownika z nazwami dla każdego klastra
cluster_labels = {}
for cluster_id in other_clusters:
    cluster_id_str = str(cluster_id)
    if cluster_id_str in cluster_names_and_descriptions:
        cluster_labels[cluster_id] = cluster_names_and_descriptions[cluster_id_str]['name']
    else:
        cluster_labels[cluster_id] = f'Grupa {cluster_id}'

# Znajdowanie przeważającego wykształcenia dla każdego klastra
cluster_edu_colors = {}
edu_color_map = {
    'Podstawowe': '#FF9999',
    'Średnie': '#99CCFF',
    'Wyższe': '#99FF99'
}

for cluster_id in other_clusters:
    cluster_df = all_df[all_df['Cluster'] == cluster_id]
    dominant_edu = cluster_df['edu_level'].mode()[0]
    cluster_edu_colors[cluster_id] = {
        'dominant_edu': dominant_edu,
        'color': edu_color_map[dominant_edu]
    }

# Tworzenie DataFrame z danymi do wykresu
comparison_df = pd.DataFrame({
    'Grupa': [cluster_labels[cluster_id] for cluster_id in cluster_sizes.index],
    'Liczba osób': cluster_sizes.values,
    'Dominujące wykształcenie': [cluster_edu_colors[cluster_id]['dominant_edu'] for cluster_id in cluster_sizes.index],
    'Kolor': [cluster_edu_colors[cluster_id]['color'] for cluster_id in cluster_sizes.index],
    'Cluster_ID': cluster_sizes.index  # Zachowujemy oryginalne ID dla późniejszego użycia
})

# Tworzenie wykresu z kolorami według wykształcenia
fig_comparison = px.bar(
    comparison_df,
    x='Grupa',
    y='Liczba osób',
    color='Dominujące wykształcenie',
    color_discrete_map=edu_color_map,
    title='Wielkość grup według dominującego wykształcenia',
    labels={'x': 'Grupa', 'y': 'Liczba osób', 'color': 'Dominujące wykształcenie'}
)

# Dodanie linii oznaczającej aktualną grupę
fig_comparison.add_vline(
    x=comparison_df[comparison_df['Cluster_ID'] == predicted_cluster_id].index[0], 
    line_dash='dash', 
    line_color='red', 
    annotation_text='Twoja grupa'
)

st.plotly_chart(fig_comparison)

# Utworzenie wykresu radarowego dla profilu grupy
categories = ['Wiek', 'Wykształcenie', 'Zwierzęta', 'Miejsce', 'Płeć']
fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=[same_cluster_df['age'].value_counts().max(),
       same_cluster_df['edu_level'].value_counts().max(),
       same_cluster_df['fav_animals'].value_counts().max(),
       same_cluster_df['fav_place'].value_counts().max(),
       same_cluster_df['gender'].value_counts().max()],
    theta=categories,
    fill='toself',
    name='Profil grupy'
))

st.header('Charakterystyka profilu grupy')
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 50]))
)
st.plotly_chart(fig_radar)


st.header('Osoby z grupy')
fig = px.histogram(same_cluster_df.sort_values('age'), x='age')
fig.update_layout(
    title='Rozkład wieku w grupie',
    xaxis_title='Wiek',
    yaxis_title='Liczba osób',
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x='edu_level')
fig.update_layout(
    title='Rozkład wykształcenia w grupie',
    xaxis_title='Wykształcenie',
    yaxis_title='Liczba osób',
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x='fav_animals')
fig.update_layout(
    title='Rozkład ulubionych zwierząt w grupie',
    xaxis_title='Ulubione zwierzęta',
    yaxis_title='Liczba osób',
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x='fav_place')
fig.update_layout(
    title='Rozkład ulubionych miejsc w grupie',
    xaxis_title='Ulubione miejsce',
    yaxis_title='Liczba osób',
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x='gender')
fig.update_layout(
    title='Rozkład płci w grupie',
    xaxis_title='Płeć',
    yaxis_title='Liczba osób',
)
st.plotly_chart(fig)
# Utworzenie wykresu radarowego dla profilu grupy
categories = ['Wiek', 'Wykształcenie', 'Zwierzęta', 'Miejsce', 'Płeć']
fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=[same_cluster_df['age'].value_counts().max(),
       same_cluster_df['edu_level'].value_counts().max(),
       same_cluster_df['fav_animals'].value_counts().max(),
       same_cluster_df['fav_place'].value_counts().max(),
       same_cluster_df['gender'].value_counts().max()],
    theta=categories,
    fill='toself',
    name='Profil grupy'
))

