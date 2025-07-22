import numpy as np
import pickle
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# --- General Config ---
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Material Design Inspired Look ---
st.markdown("""
<style>
body { background: #f5f7fa !important; }
h1, h2, h3, h4, h5 { color: #28446e; }
.material-card {
    background: #fff;
    border-radius: 18px;
    box-shadow: 0 2px 12px rgba(80,120,180,0.10);
    padding: 2.2rem 2rem 1.5rem 2rem;
    margin-bottom: 2rem;
}
.material-form-label {
    color: #667eea;
    font-weight: 500;
}
.result-card {
    background: linear-gradient(120deg, #E0EAFC 0%, #CFDEF3 100%);
    border-radius: 15px;
    padding: 1.7rem 1.5rem;
    margin-top: 1.5rem;
    box-shadow: 0 1px 6px rgba(102,126,234,0.07);
    text-align: center;
    font-size: 1.35rem;
    font-weight: 600;
}
.result-ok {
    color: #00897b;
}
.result-bad {
    color: #d32f2f;
}
.footer-tidy {
    text-align: center;
    margin-top: 2.5rem;
    color: #8a98a6;
    font-size: 0.98rem;
}
a.footer-link { color: #667eea; text-decoration: none; }
a.footer-link:hover { text-decoration: underline; }
hr { border-color: #e0e4ea; }
</style>
""", unsafe_allow_html=True)

# --- Model Loader ---
@st.cache_resource
def load_model():
    return pickle.load(open('Projects/trained_model.sav', 'rb'))

loaded_model = load_model()

# --- Prediction Function ---
def diabetes_prediction(input_data):
    try:
        input_data_float = []
        for val in input_data:
            if val == '' or val is None:
                return None, "❌ Please fill in all fields with valid numbers."
            input_data_float.append(float(val))
        input_data_as_numpy_array = np.asarray(input_data_float)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = loaded_model.predict(input_data_reshaped)
        if prediction[0] == 0:
            return 0, '🩺 You are NOT diabetic based on the analysis.'
        else:
            return 1, '⚠️ HIGH RISK: Likely diabetic, please consult a doctor.'
    except ValueError:
        return None, "❌ Please enter valid numeric values for all fields."
    except Exception as e:
        return None, f"❌ An error occurred: {str(e)}"

# --- Main UI ---
def main():
    st.markdown("<h2 style='text-align: left; margin-bottom: 0;'>🩺Diabetes Risk Predictor</h2>", unsafe_allow_html=True)
    st.markdown("<span style='color:#666;font-size:1.12rem'>A modern, ML-powered diabetes screening tool</span>", unsafe_allow_html=True)
    st.write("")

    # --- Layout ---
    sidebar, maincol, _ = st.columns([1,2.2,0.25])

    # --- Sidebar: About, Info, Guidelines ---
    with sidebar:
        st.markdown('<div class="material-card">', unsafe_allow_html=True)
        st.markdown("#### ℹ️ About")
        st.write("""
- **Model:** SVM (Support Vector Machine)
- **Dataset:** Pima Indians Diabetes
- **Real-time:** Instant ML prediction
        """)
        st.markdown("**Parameter Ranges**")
        st.caption("Pregnancies: 0–20\nGlucose: 0–200\nBP: 0–200\nSkin: 0–100\nInsulin: 0–500\nBMI: 10–67\nPedigree: 0.078–2.42\nAge: 1–120")
        st.warning("⚠️ Educational use only. Not a substitute for medical advice.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Main Content: Form and Result ---
    with maincol:
        st.markdown('<div class="material-card">', unsafe_allow_html=True)
        st.markdown("#### Enter Your Health Data")

        with st.form("diabetes_form"):
            c1, c2 = st.columns(2)
            with c1:
                Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=None, step=1, help="Number of times pregnant")
                Glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=200, value=None, step=1)
                BloodPressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=None, step=1)
                SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=None, step=1)
            with c2:
                Insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=500, value=None, step=1)
                BMI = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=67.0, value=None, step=0.1, format="%.1f")
                DiabetesPedigreeFunction = st.number_input("Pedigree Function", min_value=0.078, max_value=2.42, value=None, step=0.001, format="%.3f")
                Age = st.number_input("Age (years)", min_value=1, max_value=120, value=None, step=1)
            submit = st.form_submit_button("Assess Risk", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # --- Prediction Output ---
        if submit:
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                          Insulin, BMI, DiabetesPedigreeFunction, Age]
            with st.spinner("Analyzing your data..."):
                result, message = diabetes_prediction(input_data)
            st.markdown('<div class="result-card {}">'.format('result-bad' if result == 1 else 'result-ok' if result == 0 else ''),
                        unsafe_allow_html=True)
            if result == 0:
                st.markdown(f"{message}", unsafe_allow_html=True)
                st.info("Tips: Keep up with healthy eating, regular exercise, and annual checkups.")
            elif result == 1:
                st.markdown(f"<span style='font-size:2rem'>🚨</span><br>{message}", unsafe_allow_html=True)
                st.error("Please seek medical consultation and follow up with lab tests.")
            else:
                st.markdown(message, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Minimalist Footer ---
    st.markdown(
        """<hr>
        <div class="footer-tidy">
            Developed by <a class="footer-link" href="https://github.com/Sameer-0904" target="_blank">Sameer Prajapati</a>
            &nbsp;|&nbsp; 
            <a class="footer-link" href="mailto:sameerprajapati0904@gmail.com">Contact</a>
            <br>
            <span style="font-size:0.92rem">This tool does not provide medical advice. &copy; 2025</span>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
