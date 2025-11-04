# main.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import hashlib
from typing import List, Dict

# ---------- Config / filenames ----------
MODEL_KERAS = "trained_model.keras"
MODEL_H5 = "trained_model.h5"
DATA_JSON = "symptoms_dataset.json"   # place your provided JSON here
USERS_JSON = "users.json"             # simple user store for auth

SUPPORTED_LANGS = {
    "en": "English",
    "hi": "Hindi",
    "ur": "Urdu",
    "ta": "Tamil",
    "mr": "Marathi",
    "bn": "Bengali",
    "te": "Telugu",
    "kn": "Kannada"
}

# ---------- Utilities ----------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def ensure_users_file():
    if not os.path.exists(USERS_JSON):
        # create default users: admin/admin123 and farmer/farmer123
        users = {
            "admin": {"password": hash_password("admin123"), "role": "admin"},
            "farmer": {"password": hash_password("farmer123"), "role": "farmer"}
        }
        with open(USERS_JSON, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
ensure_users_file()

def load_users() -> dict:
    with open(USERS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users: dict):
    with open(USERS_JSON, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def load_dataset() -> List[dict]:
    if not os.path.exists(DATA_JSON):
        # create empty dataset placeholder so admin can add entries easily
        with open(DATA_JSON, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def save_dataset(data: List[dict]):
    with open(DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def find_disease_by_id(data: List[dict], disease_id: str):
    for d in data:
        if d.get("disease_id") == disease_id or d.get("disease") == disease_id:
            return d
    return None

def search_diseases(data: List[dict], query: str, lang: str = "en") -> List[dict]:
    q = query.lower().strip()
    results = []
    for d in data:
        # search disease name
        if q in d.get("disease", "").lower():
            results.append(d); continue
        # search disease_id
        if q in d.get("disease_id", "").lower():
            results.append(d); continue
        # search symptoms and treatment in all languages or in chosen language
        # checking chosen language first then fallback to english
        fields_to_check = []
        if lang and d.get("symptoms", {}).get(lang):
            fields_to_check.append(d["symptoms"].get(lang, ""))
        else:
            fields_to_check.extend(d.get("symptoms", {}).values())
        if any(q in (str(f).lower()) for f in fields_to_check):
            results.append(d)
            continue
        # also search treatments
        if lang and d.get("treatment", {}).get(lang):
            if q in str(d["treatment"].get(lang, "")).lower():
                results.append(d); continue
        else:
            if any(q in str(t).lower() for t in d.get("treatment", {}).values()):
                results.append(d); continue
    return results

# ---------- Model helpers ----------
def load_model_file():
    if os.path.exists(MODEL_KERAS):
        try:
            return tf.keras.models.load_model(MODEL_KERAS)
        except Exception:
            pass
    if os.path.exists(MODEL_H5):
        return tf.keras.models.load_model(MODEL_H5)
    return None

_model_cache = None
def get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model_file()
    return _model_cache

def model_predict_index(uploaded_file):
    model = get_model()
    if model is None:
        raise FileNotFoundError("No model found (trained_model.keras or trained_model.h5).")
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((128, 128))
    arr = tf.keras.preprocessing.image.img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    score = float(np.max(preds))
    return idx, score

# ---------- Streamlit App ----------
st.set_page_config(page_title="AgroBot – Plant Disease + Chatbot", layout="wide")
st.sidebar.title("AgroBot Dashboard")

# Authentication area
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None

def do_login(username: str, password: str):
    users = load_users()
    user = users.get(username)
    if user and user.get("password") == hash_password(password):
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.role = user.get("role", "farmer")
        st.success(f"Logged in as {username} ({st.session_state.role})")
        return True
    st.error("Invalid username or password.")
    return False

def do_logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
    st.experimental_rerun()

# Sidebar: Login / Logout
if not st.session_state.logged_in:
    st.sidebar.subheader("Login")
    u = st.sidebar.text_input("Username")
    p = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        do_login(u.strip(), p.strip())
    st.sidebar.markdown("**Default demo accounts:**\n\n- admin / admin123  \n- farmer / farmer123")
else:
    st.sidebar.markdown(f"**User:** {st.session_state.username}  \n**Role:** {st.session_state.role}")
    if st.sidebar.button("Logout"):
        do_logout()

# Main navigation (Home / About / Disease Recognition / Chatbot / Admin)
page = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Chatbot", "Admin"])

# ---------- Home ----------
if page == "Home":
    st.header("AGROBOT : UNIVERSAL AI BASED AGRICULTURAL ASSISTANT")
    st.image("home_page.jpeg" if os.path.exists("home_page.jpeg") else None, use_column_width=True)
    st.markdown("""
    Welcome to AgroBot — you can:
    - Upload an image to detect disease (Disease Recognition).
    - Ask the multilingual chatbot about symptoms and treatments (Chatbot).
    - Admins can manage the dataset (Admin).
    """)
# ---------- About ----------
elif page == "About":
    st.header("About")
    st.markdown("""
    This application merges image-based plant disease recognition with a multilingual chatbot backed by a JSON dataset.
    **Languages supported:** English, Hindi, Urdu, Tamil, Marathi, Bengali, Telugu, Kannada.
    Admins can modify the disease and treatment JSON used by the chatbot.
    """)

# ---------- Disease Recognition ----------
elif page == "Disease Recognition":
    st.header("Disease Recognition")
    uploaded_file = st.file_uploader("Choose an Image", type=["png","jpg","jpeg"])
    show = st.button("Show Image")
    if uploaded_file is not None and show:
        st.image(uploaded_file, use_column_width=True)
    if st.button("Predict"):
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Predicting..."):
                try:
                    idx, score = model_predict_index(uploaded_file)
                    # class list - keep same as you had (shortened here if needed)
                    class_name = ['Apple___Apple_scab',
                        'Apple___Black_rot',
                        'Apple___Cedar_apple_rust',
                        'Apple___healthy',
                        'Blueberry___healthy',
                        'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy',
                        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_',
                        'Corn_(maize)___Northern_Leaf_Blight',
                        'Corn_(maize)___healthy',
                        'Grape___Black_rot',
                        'Grape___Esca_(Black_Measles)',
                        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Grape___healthy',
                        'Orange___Haunglongbing_(Citrus_greening)',
                        'Peach___Bacterial_spot',
                        'Peach___healthy',
                        'Pepper,_bell___Bacterial_spot',
                        'Pepper,_bell___healthy',
                        'Potato___Early_blight',
                        'Potato___Late_blight',
                        'Potato___healthy',
                        'Raspberry___healthy',
                        'Soybean___healthy',
                        'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch',
                        'Strawberry___healthy',
                        'Tomato___Bacterial_spot',
                        'Tomato___Early_blight',
                        'Tomato___Late_blight',
                        'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot',
                        'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot',
                        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                        'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
                    if idx < len(class_name):
                        predicted = class_name[idx]
                    else:
                        predicted = f"Class_{idx}"
                    st.success(f"Predicted: **{predicted}** (confidence {score:.2f})")
                    # Save predicted in session for chatbot convenience
                    st.session_state.last_prediction = {"disease_id": predicted, "confidence": score}
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ---------- Chatbot ----------
elif page == "Chatbot":
    st.header("AgroBot — Multilingual Chatbot")
    language = st.selectbox("Choose language", list(SUPPORTED_LANGS.keys()), format_func=lambda k: SUPPORTED_LANGS[k])
    data = load_dataset()
    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("### Ask AgroBot")
        st.markdown("You can: (1) type disease name, (2) type symptom keywords, or (3) use image prediction (if you just predicted).")
        query = st.text_input("Ask question (e.g. 'apple scab' or 'brown spots on leaves')")

        use_pred = False
        if st.session_state.get("last_prediction"):
            pred = st.session_state.last_prediction
            if st.button("Use last image prediction in chatbot"):
                use_pred = True
                # Will use predicted disease_id below

        if st.button("Get Answer"):
            if use_pred:
                disease_id = st.session_state["last_prediction"]["disease_id"]
                d = find_disease_by_id(data, disease_id)
                if d is None:
                    st.warning("Predicted disease not found in dataset. Try searching by keywords.")
                else:
                    thresh = d.get("confidence_threshold", 0.0)
                    st.markdown(f"**Disease:** {d.get('disease')}  \n**Confidence threshold stored:** {thresh}")
                    sym = d.get("symptoms", {}).get(language) or d.get("symptoms", {}).get("en") or "Not available"
                    treat = d.get("treatment", {}).get(language) or d.get("treatment", {}).get("en") or "Not available"
                    st.markdown(f"**Symptoms ({SUPPORTED_LANGS.get(language)}):**\n{sym}")
                    st.markdown(f"**Treatment ({SUPPORTED_LANGS.get(language)}):**\n{treat}")
            else:
                if not query.strip():
                    st.warning("Please type a question or use prediction.")
                else:
                    results = search_diseases(data, query, lang=language)
                    if not results:
                        st.info("No matches found in dataset. Try different keywords or check spelling.")
                    else:
                        st.success(f"Found {len(results)} result(s). Showing top 5.")
                        for d in results[:5]:
                            st.markdown("---")
                            st.markdown(f"**Disease id:** {d.get('disease_id')}")
                            st.markdown(f"**Disease name:** {d.get('disease')}")
                            sym = d.get("symptoms", {}).get(language) or d.get("symptoms", {}).get("en") or "Not available"
                            treat = d.get("treatment", {}).get(language) or d.get("treatment", {}).get("en") or "Not available"
                            st.markdown(f"**Symptoms ({SUPPORTED_LANGS.get(language)}):** {sym}")
                            st.markdown(f"**Treatment ({SUPPORTED_LANGS.get(language)}):** {treat}")

    with col2:
        st.markdown("### Quick Tips")
        st.markdown("- Use short keywords.  \n- You can ask in any supported language, but searching works best with English or disease_ids.")
        st.markdown("### Dataset Info")
        st.write(f"Total disease entries: **{len(load_dataset())}**")

# ---------- Admin (CRUD) ----------
elif page == "Admin":
    st.header("Admin Panel — Manage Disease Dataset")
    if not st.session_state.logged_in or st.session_state.role != "admin":
        st.warning("You must be logged in as an admin to access this page.")
    else:
        data = load_dataset()
        st.subheader("Existing entries")
        expander = st.expander("View dataset JSON")
        with expander:
            st.write(data)

        # Select a disease to edit/delete
        ids = [d.get("disease_id") for d in data]
        sel = st.selectbox("Select disease to edit", ["-- Add new --"] + ids)
        if sel != "-- Add new --":
            d = find_disease_by_id(data, sel)
            if d:
                st.text_input("disease_id", value=d.get("disease_id"), key="edit_id")
                st.text_input("disease (display name)", value=d.get("disease"), key="edit_name")
                # Language-wise symptoms and treatment
                st.markdown("**Symptoms (by language)**")
                for lang in SUPPORTED_LANGS:
                    st.text_area(f"Symptoms ({lang})", value=d.get("symptoms", {}).get(lang, ""), key=f"sym_{lang}")
                st.markdown("**Treatment (by language)**")
                for lang in SUPPORTED_LANGS:
                    st.text_area(f"Treatment ({lang})", value=d.get("treatment", {}).get(lang, ""), key=f"tr_{lang}")
                ct = st.number_input("confidence_threshold", value=float(d.get("confidence_threshold", 0.25)), step=0.01, key="edit_ct")
                if st.button("Save changes"):
                    # apply edits
                    new_id = st.session_state.pop("edit_id")
                    new_name = st.session_state.pop("edit_name")
                    # build new entry
                    new_entry = {
                        "disease_id": new_id,
                        "disease": new_name,
                        "symptoms": {},
                        "treatment": {},
                        "confidence_threshold": float(ct)
                    }
                    for lang in SUPPORTED_LANGS:
                        new_entry["symptoms"][lang] = st.session_state.pop(f"sym_{lang}", "")
                        new_entry["treatment"][lang] = st.session_state.pop(f"tr_{lang}", "")
                    # replace in data
                    for i, old in enumerate(data):
                        if old.get("disease_id") == sel:
                            data[i] = new_entry
                            save_dataset(data)
                            st.success("Entry updated.")
                            break
                if st.button("Delete entry"):
                    data = [x for x in data if x.get("disease_id") != sel]
                    save_dataset(data)
                    st.success("Entry deleted.")
        else:
            st.markdown("### Add new disease entry")
            add_id = st.text_input("disease_id (unique)", key="new_id")
            add_name = st.text_input("disease (display name)", key="new_name")
            st.markdown("**Symptoms (by language)**")
            for lang in SUPPORTED_LANGS:
                st.text_area(f"Symptoms ({lang})", key=f"new_sym_{lang}")
            st.markdown("**Treatment (by language)**")
            for lang in SUPPORTED_LANGS:
                st.text_area(f"Treatment ({lang})", key=f"new_tr_{lang}")
            add_ct = st.number_input("confidence_threshold", value=0.25, step=0.01, key="new_ct")
            if st.button("Add entry"):
                if not add_id.strip():
                    st.error("Provide a unique disease_id.")
                else:
                    # build entry
                    entry = {
                        "disease_id": add_id.strip(),
                        "disease": add_name.strip() if add_name.strip() else add_id.strip(),
                        "symptoms": {},
                        "treatment": {},
                        "confidence_threshold": float(add_ct)
                    }
                    for lang in SUPPORTED_LANGS:
                        entry["symptoms"][lang] = st.session_state.pop(f"new_sym_{lang}", "")
                        entry["treatment"][lang] = st.session_state.pop(f"new_tr_{lang}", "")
                    data.append(entry)
                    save_dataset(data)
                    st.success("New disease entry added.")

        # Admin: manage users
        st.markdown("---")
        st.subheader("Manage users")
        users = load_users()
        st.write(users)
        new_user = st.text_input("New username")
        new_pass = st.text_input("New password", type="password")
        new_role = st.selectbox("Role", ["farmer", "admin"])
        if st.button("Add user"):
            if not new_user.strip() or not new_pass:
                st.error("Enter username and password.")
            else:
                users[new_user.strip()] = {"password": hash_password(new_pass.strip()), "role": new_role}
                save_users(users)
                st.success("User added.")
                st.experimental_rerun()

# ---------- Fallback ----------
else:
    st.write("Select a page from the sidebar.")
