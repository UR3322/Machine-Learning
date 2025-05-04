# app.py - Universal ML Platform with Minecraft Theme
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configure page
st.set_page_config(
    page_title="ML CRAFT - By Samad Kiani",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Minecraft Theme
st.markdown("""
<style>
    /* Main App Background */
    body, .stApp {
        background-color: #5b7c3d;  /* Minecraft grass green */
        background-image: url('https://i.imgur.com/JGQhQaT.png');
        background-size: 300px;
        color: #ffffff;
        font-family: 'Minecraft', sans-serif;
    }
    
    /* Import Minecraft font */
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
    
    /* Main Content Container */
    .main {
        background-color: #8b8b8b;  /* Minecraft stone gray */
        padding: 2rem;
        border-radius: 0px;
        border: 4px solid #3a3a3a;  /* Minecraft dark stone border */
        box-shadow: 8px 8px 0px #3a3a3a;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffaa00 !important;  /* Minecraft gold */
        font-family: 'Press Start 2P', cursive;
        text-shadow: 3px 3px 0px #3a3a3a;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #5b7c3d !important;  /* Grass green */
        border-right: 4px solid #3a3a3a;
        font-family: 'Press Start 2P', cursive;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #b85233;  /* Minecraft wood brown */
        color: white;
        border-radius: 0px;
        border: 3px solid #3a3a3a;
        padding: 0.5rem 1rem;
        font-family: 'Press Start 2P', cursive;
        font-size: 12px;
        box-shadow: 4px 4px 0px #3a3a3a;
        transition: all 0.1s ease;
    }
    
    .stButton>button:hover {
        background-color: #d87f33;  /* Lighter wood */
        transform: translate(2px, 2px);
        box-shadow: 2px 2px 0px #3a3a3a;
    }
    
    /* Download Button */
    .stDownloadButton>button {
        background-color: #5b7c3d;  /* Grass green */
        color: white;
    }
    
    /* File Uploader */
    .stFileUploader>div>div {
        background-color: #8b8b8b;
        border: 3px dashed #3a3a3a;
        border-radius: 0px;
    }
    
    /* Expanders */
    .st-expander {
        background-color: #8b8b8b;
        border: 3px solid #3a3a3a;
        border-radius: 0px;
    }
    
    .st-expanderHeader {
        color: #ffaa00 !important;
        font-family: 'Press Start 2P', cursive;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #8b8b8b !important;
        color: #ffffff !important;
        border: 3px solid #3a3a3a !important;
    }
    
    /* Input Widgets */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select,
    .stMultiselect>div>div>div {
        background-color: #8b8b8b;
        color: #ffffff;
        border: 3px solid #3a3a3a;
        border-radius: 0px;
        font-family: 'Press Start 2P', cursive;
    }
    
    /* Slider */
    .stSlider>div>div>div>div {
        background-color: #ffaa00;  /* Gold */
    }
    
    /* Tabs */
    .stTabs>div>div>div>button {
        background-color: #8b8b8b;
        color: #ffffff;
        border: 3px solid #3a3a3a;
        border-radius: 0px;
        font-family: 'Press Start 2P', cursive;
    }
    
    .stTabs>div>div>div>button[aria-selected="true"] {
        color: #ffaa00;
        background-color: #5b7c3d;
    }
    
    /* Progress Bar */
    .stProgress>div>div>div>div {
        background-color: #ffaa00;
    }
    
    /* Alerts */
    .stAlert .st-at {
        background-color: #8b8b8b !important;
        border: 3px solid #3a3a3a !important;
        color: #ffffff !important;
        border-radius: 0px;
    }
    
    /* Metric Cards */
    .stMetric {
        background-color: #8b8b8b;
        border: 3px solid #3a3a3a;
        border-radius: 0px;
        padding: 15px;
    }
    
    /* Plotly Chart Background */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly div {
        background-color: #8b8b8b !important;
    }
    
    /* Block pixel effect */
    .block-pixel {
        image-rendering: pixelated;
        image-rendering: -moz-crisp-edges;
        image-rendering: crisp-edges;
    }
</style>
""", unsafe_allow_html=True)

# Main Function
def main():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.title("⛏️ ML CRAFT")
    st.markdown("---")
    
    # Session state initialization
    session_defaults = {
        'data': None, 'model': None, 'features': [], 'target': None,
        'steps': {'loaded': False, 'processed': False, 'trained': False},
        'predictions': None
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Crafting Table")
        uploaded_file = st.file_uploader("Upload Dataset:", type=["csv", "xlsx"])
        
        st.markdown("---")
        st.header("🧠 Redstone Settings")
        model_type = st.selectbox("Select Model:", ["Linear Regression", "Random Forest"])
        test_size = st.slider("Test Size Ratio:", 0.1, 0.5, 0.2)
        st.button("Reset World", on_click=lambda: st.session_state.clear())

    # Step 1: Data Upload
    st.header("1. Mine Your Data")
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) < 2:
                st.error("🧨 You need at least 2 numeric ores to craft!")
                return
                
            st.session_state.data = df
            st.session_state.steps['loaded'] = True
            st.success(f"✅ Successfully mined {len(df)} blocks of data!")
            
            st.write("### Chest Preview:")
            st.dataframe(df.head().style.format("{:.2f}", subset=numeric_cols), height=250)
            
            with st.expander("🔍 Select Ores & Target"):
                all_cols = df.columns.tolist()
                target = st.selectbox("Select Target Block:", numeric_cols, index=len(numeric_cols)-1)
                default_features = [col for col in numeric_cols if col != target][:3]
                features = st.multiselect("Select Ores:", numeric_cols, default=default_features)
                
                if st.button("Craft Selection"):
                    if len(features) < 1:
                        st.error("🧨 You need at least one ore to craft!")
                    elif target in features:
                        st.error("🧨 Target block can't be used as ore!")
                    else:
                        st.session_state.features = features
                        st.session_state.target = target
                        st.session_state.steps['processed'] = True
                        st.success("⚡ Crafting successful!")
            
        except Exception as e:
            st.error(f"💥 Mining error: {str(e)}")
    else:
        st.markdown("""
        ### How to Play:
        1. ⛏️ Upload any CSV or Excel file with numeric data  
        2. 🎯 Select target block (what you want to predict)  
        3. ⚒️ Choose ores (variables used for prediction)  
        4. 🧙 The system will automatically craft the rest  
        """)

    # Step 2: Data Analysis
    if st.session_state.steps['processed']:
        st.header("2. Smelt Your Ores")
        df = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Ore-Target Relationships")
            selected_feature = st.selectbox("Select ore to smelt:", features)
            fig = px.scatter(df, x=selected_feature, y=target, trendline="ols", height=400)
            fig.update_layout({
                'plot_bgcolor': '#8b8b8b',
                'paper_bgcolor': '#8b8b8b',
                'font': {'color': '#ffffff'}
            })
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("### Ore Combinations")
            corr_matrix = df[features + [target]].corr()
            fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='Oranges', aspect="auto")
            fig.update_layout({
                'plot_bgcolor': '#8b8b8b',
                'paper_bgcolor': '#8b8b8b',
                'font': {'color': '#ffffff'}
            })
            st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🚀 Enchant Model"):
            st.session_state.steps['ready_for_model'] = True

    # Step 3: Model Training
    if st.session_state.steps.get('ready_for_model'):
        st.header("3. Enchanting Table")
        df = st.session_state.data
        features = st.session_state.features
        target = st.session_state.target
        
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LinearRegression() if model_type == "Linear Regression" else RandomForestRegressor(n_estimators=100, random_state=42)
        
        with st.spinner(f"⚡ Enchanting {model_type}..."):
            model.fit(X_train_scaled, y_train)
            st.session_state.model = model
            st.session_state.steps['trained'] = True
            
            y_pred = model.predict(X_test_scaled)
            st.session_state.predictions = {'y_test': y_test, 'y_pred': y_pred, 'X_test': X_test}
            st.success("✨ Enchantment successful!")
            st.balloons()

    # Step 4: Evaluation
    if st.session_state.steps.get('trained'):
        st.header("4. Potion Effects")
        predictions = st.session_state.predictions
        y_test = predictions['y_test']
        y_pred = predictions['y_pred']
        X_test = predictions['X_test']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mining Error", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}", help="Lower is better")
        with col2:
            st.metric("Crafting Power", f"{r2_score(y_test, y_pred):.2f}", help="1.0 is perfect")
        
        st.write("### Actual vs Enchanted")
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results.index, 
            y=results['Actual'], 
            name='Actual', 
            mode='markers', 
            marker=dict(color='#5b7c3d', size=10)
        ))
        fig.add_trace(go.Scatter(
            x=results.index, 
            y=results['Predicted'], 
            name='Predicted', 
            mode='markers', 
            marker=dict(color='#ffaa00', size=10)
        ))
        fig.update_layout(
            xaxis_title="Block Index", 
            yaxis_title="Value", 
            height=500,
            plot_bgcolor='#8b8b8b',
            paper_bgcolor='#8b8b8b',
            font=dict(color='#ffffff')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if model_type == "Random Forest":
            st.write("### Ore Value")
            importance = pd.DataFrame({'Ore': st.session_state.features, 'Value': st.session_state.model.feature_importances_})
            importance = importance.sort_values('Value', ascending=False)
            fig = px.bar(
                importance, 
                x='Value', 
                y='Ore', 
                orientation='h', 
                color='Value', 
                color_continuous_scale='Oranges'
            )
            fig.update_layout({
                'plot_bgcolor': '#8b8b8b',
                'paper_bgcolor': '#8b8b8b',
                'font': {'color': '#ffffff'}
            })
            st.plotly_chart(fig, use_container_width=True)
        
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Enchantment", csv, "minecraft_predictions.csv", "text/csv")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()