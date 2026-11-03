import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Expand layout to use the full monitor screen
st.set_page_config(page_title="VCT Analytics Portal", layout="wide", page_icon="🏆")

@st.cache_resource
def load_assets():
    xgb_model = joblib.load('saved_models/xgboost_vct_model.pkl')
    assets = joblib.load('saved_models/encoding_assets.pkl')
    xgb_model.set_params(device='cpu')
    return xgb_model, assets

model, encoding_assets = load_assets()
mlb_a = encoding_assets['mlb_a']
mlb_b = encoding_assets['mlb_b']
map_columns = encoding_assets['map_columns']

official_agents = [
    'astra', 'breach', 'brimstone', 'chamber', 'clove', 'cypher', 
    'deadlock', 'fade', 'gekko', 'harbor', 'iso', 'jett', 'kayo', 
    'killjoy', 'miks', 'neon', 'omen', 'phoenix', 'raze', 'reyna', 'sage', 
    'skye', 'sova', 'tejo', 'viper', 'vyse', 'waylay', 'yoru'
]
available_agents = sorted([agent for agent in mlb_a.classes_ if agent in official_agents])
available_maps = [m.replace('Map_', '') for m in map_columns]

# 2. Dynamic Image Renderer
def render_image(name, category):
    file_path = f"assets/{category}/{name.lower()}.png"
    if os.path.exists(file_path):
        st.image(file_path, caption=name.capitalize(), use_container_width=True)
    else:
        html_card = f"""
        <div style='text-align: center; padding: 15px 5px; background-color: #2b2b36; 
                    border-radius: 8px; border: 1px solid #4a4a5a; margin-bottom: 15px;'>
            <h5 style='margin:0; color: #ffffff;'>{name.capitalize()}</h5>
        </div>
        """
        st.markdown(html_card, unsafe_allow_html=True)

# 3. Portal Header & Navigation Tabs
st.title("🏆 VCT Valorant Analytics Portal")
st.markdown("Predict match outcomes and analyze historical meta data from the Valorant Champions Tour (2021-2026).")

tab1, tab2 = st.tabs(["🎮 Match Predictor", "📊 Meta Dashboard"])

# ==========================================
# TAB 1: MATCH PREDICTOR (Kode Aslimu)
# ==========================================
with tab1:
    # Sidebar for Map Selection (Akan tetap muncul di samping)
    with st.sidebar:
        st.header("🗺️ Battlefield")
        selected_map = st.selectbox("Select Map", available_maps)
        st.divider()
        render_image(selected_map, "maps")

    # Main Area for Draft Compositions
    col_team_a, col_spacing, col_team_b = st.columns([1, 0.1, 1])

    with col_team_a:
        st.header("🔵 Team A Draft")
        team_a_picks = st.multiselect("Select exactly 5 agents", available_agents, max_selections=5, key="team_a")
        
        if team_a_picks:
            slots_a = st.columns(5)
            for i, agent in enumerate(team_a_picks):
                with slots_a[i]:
                    render_image(agent, "agents")

    with col_team_b:
        st.header("🔴 Team B Draft")
        team_b_picks = st.multiselect("Select exactly 5 agents", available_agents, max_selections=5, key="team_b")
        
        if team_b_picks:
            slots_b = st.columns(5)
            for i, agent in enumerate(team_b_picks):
                with slots_b[i]:
                    render_image(agent, "agents")

    st.divider()
    st.info("💡 **Note:** While agent compositions provide significant strategic advantages, Valorant is fundamentally a tactical FPS. This AI model predicts outcomes based on historical meta, but ultimately, mechanical skill, raw aim, and in-game execution will decide the true winner on the server.")

    # Inference Logic
    if st.button("Predict Match Outcome", use_container_width=True, type="primary"):
        if len(team_a_picks) != 5 or len(team_b_picks) != 5:
            st.error("⚠️ Invalid draft: Both teams must select exactly 5 agents before simulating.")
        else:
            team_a_custom = pd.DataFrame(mlb_a.transform([team_a_picks]), columns=['Team_A_' + col for col in mlb_a.classes_])
            team_b_custom = pd.DataFrame(mlb_b.transform([team_b_picks]), columns=['Team_B_' + col for col in mlb_b.classes_])
            
            map_custom = pd.DataFrame(0, index=[0], columns=map_columns)
            map_column_name = f"Map_{selected_map}"
            if map_column_name in map_custom.columns:
                map_custom[map_column_name] = 1
                
            X_custom = pd.concat([map_custom, team_a_custom, team_b_custom], axis=1)
            
            probabilities = model.predict_proba(X_custom)[0]
            team_b_win_prob = probabilities[0] * 100
            team_a_win_prob = probabilities[1] * 100
            
            res_col1, res_col2 = st.columns(2)
            res_col1.metric(label="🔵 Team A Win Probability", value=f"{team_a_win_prob:.2f}%")
            res_col2.metric(label="🔴 Team B Win Probability", value=f"{team_b_win_prob:.2f}%")

# ==========================================
# TAB 2: META & SYNERGY DASHBOARD
# ==========================================
with tab2:
    st.header("📊 VCT Meta Analysis")
    st.write("Explore agent pick rates and win rates across the professional VCT circuit.")
    
    # 1. Prepare dummy data for ALL official agents
    np.random.seed(42)
    dummy_meta = pd.DataFrame({
        'Agent': [agent.capitalize() for agent in official_agents],
        'Pick Rate (%)': np.random.uniform(5, 80, len(official_agents)).round(1),
        'Win Rate (%)': np.random.uniform(40, 60, len(official_agents)).round(1)
    })

    # 2. Dropdown Filter for Sorting
    sort_option = st.selectbox(
        "Sort Data By:",
        ["Highest Pick Rate", "Lowest Pick Rate", "Highest Win Rate", "Lowest Win Rate"]
    )

    # Apply Sorting based on dropdown selection
    if sort_option == "Highest Pick Rate":
        dummy_meta = dummy_meta.sort_values(by="Pick Rate (%)", ascending=False)
    elif sort_option == "Lowest Pick Rate":
        dummy_meta = dummy_meta.sort_values(by="Pick Rate (%)", ascending=True)
    elif sort_option == "Highest Win Rate":
        dummy_meta = dummy_meta.sort_values(by="Win Rate (%)", ascending=False)
    elif sort_option == "Lowest Win Rate":
        dummy_meta = dummy_meta.sort_values(by="Win Rate (%)", ascending=True)

    # Clean up index to show neat numbering starting from 1
    dummy_meta = dummy_meta.reset_index(drop=True)
    dummy_meta.index += 1 

    st.divider()

    # 3. Display as a neat, formatted list with visual bars
    st.dataframe(
        dummy_meta,
        use_container_width=True,
        column_config={
            "Agent": st.column_config.TextColumn(
                "Agent Name",
                width="medium"
            ),
            "Pick Rate (%)": st.column_config.ProgressColumn(
                "Pick Rate (%)",
                help="Agent selection frequency",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Win Rate (%)": st.column_config.ProgressColumn(
                "Win Rate (%)",
                help="Agent match win percentage",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            )
        }
    )