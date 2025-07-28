import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import streamlit.components.v1 as components

st.set_page_config(page_title="vermeg", layout="wide")

# ---- Fonction de connexion ----
def login():
    st.title("🔐 Connexion")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if username == "YoldezDerouiche" and password == "1234":
            st.session_state.logged_in = True
            st.success("Connexion réussie !")
            st.rerun()    # ← Utilise cette ligne
        else:
            st.error("Identifiants incorrects.")

# ---- Initialisation session ----
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---- Affichage ----
if not st.session_state.logged_in:
    login()
    st.stop()   # ARRÊTE L’EXÉCUTION : on n’affiche RIEN d’autre
else:
    # (Optionnel) Masquer l'en-tête Streamlit et autres éléments natifs
    st.markdown("""
        <style>
        [data-testid="stHeader"], [data-testid="stToolbar"], footer, #MainMenu {
            display: none !important;
        }
        .block-container {
            padding: 0;
            margin: 0;
        }
        iframe {
            border: none;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # ---- Sidebar ----
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Navigation", ["Dashboard Power BI", "Prédiction LSTM", "Déconnexion"])

    # ---- Déconnexion ----
    if page == "Déconnexion":
        st.session_state.logged_in = False
        st.rerun()

    # ---- Power BI ----
    if page == "Dashboard Power BI":
        st.title("📊 Dashboard Power BI")
        components.html("""
            <iframe src="https://app.powerbi.com/view?r=eyJrIjoiYTY4OWE0YzUtMDg3MS00N2ExLTkyYjktNWMyM2FhNDlkYTdhIiwidCI6ImRiZDY2NjRkLTRlYjktNDZlYi05OWQ4LTVjNDNiYTE1M2M2MSIsImMiOjl9&pageName=ReportSection20eca5fb43c95415487b"
                    allowfullscreen="true"
                    style="width:100vw; height:100vh;">
            </iframe>
        """, height=1000, scrolling=False)

    # ---- Prédiction LSTM ----
    elif page == "Prédiction LSTM":
        st.title("🔮 Prédiction du délai de résolution d'un ticket support (LSTM)")
        with open('label_encoder_priority.pkl', 'rb') as f:
            le_priority = pickle.load(f)
        with open('label_encoder_platform.pkl', 'rb') as f:
            le_platform = pickle.load(f)
        with open('label_encoder_status.pkl', 'rb') as f:
            le_status = pickle.load(f)
        with open('scaler_resolution.pkl', 'rb') as f:
            scaler = pickle.load(f)
        model = load_model('tickets_lstm_resolution_delay.h5', compile=False)

        priority = st.selectbox("Priorité", le_priority.classes_)
        platform = st.selectbox("Plateforme", le_platform.classes_)
        status = st.selectbox("Statut", le_status.classes_)
        hour = st.slider("Heure de création du ticket", 0, 23, 9)
        is_weekend = st.radio("Est-ce le week-end ?", ("Non", "Oui"))

        priority_enc = le_priority.transform([priority])[0]
        platform_enc = le_platform.transform([platform])[0]
        status_enc = le_status.transform([status])[0]
        is_weekend_val = 1 if is_weekend == "Oui" else 0

        input_data = np.array([[priority_enc, platform_enc, status_enc, hour, is_weekend_val]], dtype=np.float32)
        input_scaled = scaler.transform(np.array([[hour, 0]]))
        input_data[:, 3] = input_scaled[:, 0]

        sequence_length = 10
        X_pred = np.tile(input_data, (sequence_length, 1)).reshape(1, sequence_length, input_data.shape[1])

        y_pred_scaled = model.predict(X_pred)
        resolution_delay_hour = scaler.inverse_transform(np.array([[hour, y_pred_scaled[0, 0]]]))[0, 1]
        resolution_delay_days = resolution_delay_hour / 24

        st.write(f"### ⏱️ Délai de résolution prédit : *{resolution_delay_hour:.2f} heures*")
        st.write(f"### 🗓️ Délai de résolution prédit : *{resolution_delay_days:.2f} jours*")
