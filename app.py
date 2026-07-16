import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import shap

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# =====================================
# Load Files
# =====================================
model = joblib.load("artifacts/model.pkl")
encoder = joblib.load("artifacts/encoder.pkl")
explainer = joblib.load("artifacts/explainer.pkl")

with open("artifacts/metrics.json") as file:
    metrics = json.load(file)

data = pd.read_csv("artifacts/ai4i2020.csv")
X_test = pd.read_csv("artifacts/X_test.csv")
y_test = pd.read_csv("artifacts/y_test.csv")

# =====================================
# Features
# =====================================
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# =====================================
# Sidebar
# =====================================
st.sidebar.title("Predictive Maintenance")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Model Performance",
        "Prediction",
        "Explainability",
        "Data Explorer"
    ]
)

# =====================================
# Dashboard
# =====================================
if page == "Dashboard":

    st.title("📊 Predictive Maintenance Dashboard")

    df = data.copy()

    # Encode Machine Type
    if df["Type"].dtype == "object":
        df["Type"] = encoder.transform(df["Type"])

    # Prediction
    probability = model.predict_proba(df[features])[:, 1]
    df["Failure Probability"] = probability

    # Risk Level
    def risk(x):
        if x > 0.6:
            return "High"
        elif x > 0.3:
            return "Medium"
        else:
            return "Low"

    df["Risk"] = df["Failure Probability"].apply(risk)

    # KPI
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Machines", len(df))
    c2.metric("High Risk", len(df[df["Risk"] == "High"]))
    c3.metric("Failures", int(df["Machine failure"].sum()))
    c4.metric("Failure Rate", f"{df['Machine failure'].mean()*100:.2f}%")

    st.divider()

    # Histogram
    st.subheader("Failure Probability Distribution")

    fig = px.histogram(
        df,
        x="Failure Probability",
        color="Risk",
        nbins=30
    )

    st.plotly_chart(fig, use_container_width=True)

    # Pie Chart
    st.subheader("Machine Type Distribution")

    temp = data.copy()

    fig = px.pie(
        temp,
        names="Type"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Box Plot
    st.subheader("Torque by Machine Type")

    fig = px.box(
        temp,
        x="Type",
        y="Torque [Nm]",
        color="Type"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Top Risk Machines
    st.subheader("Top 10 High Risk Machines")

    top = df.sort_values(
        "Failure Probability",
        ascending=False
    ).head(10)

    st.dataframe(
        top[
            [
                "UDI",
                "Type",
                "Failure Probability",
                "Machine failure"
            ]
        ],
        use_container_width=True
    )

    # =====================================
# Model Performance
# =====================================
elif page == "Model Performance":

    st.title("📈 Model Performance")

    result = metrics["metrics"]

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Accuracy", f"{result['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{result['precision']*100:.2f}%")
    c3.metric("Recall", f"{result['recall']*100:.2f}%")
    c4.metric("F1 Score", f"{result['f1']*100:.2f}%")
    c5.metric("ROC-AUC", f"{result['roc_auc']:.3f}")

    st.divider()

    st.subheader("Confusion Matrix")

    cm = np.array(metrics["confusion_matrix"])

    fig = px.imshow(
        cm,
        text_auto=True,
        x=["No Failure","Failure"],
        y=["No Failure","Failure"],
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ROC Curve")

    fpr = metrics["roc_curve"]["fpr"]
    tpr = metrics["roc_curve"]["tpr"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="ROC Curve"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0,1],
            y=[0,1],
            mode="lines",
            name="Random"
        )
    )

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Prediction Probability")

    probability=model.predict_proba(X_test[features])[:,1]

    fig=px.histogram(
        x=probability,
        nbins=30
    )

    st.plotly_chart(fig,use_container_width=True)

    # =====================================
# Prediction
# =====================================
elif page=="Prediction":

    st.title("🔮 Machine Failure Prediction")

    machine=st.selectbox(
        "Machine Type",
        ["L","M","H"]
    )

    air=st.number_input(
        "Air Temperature (K)",
        value=300.0
    )

    process=st.number_input(
        "Process Temperature (K)",
        value=310.0
    )

    speed=st.number_input(
        "Rotational Speed (RPM)",
        value=1500.0
    )

    torque=st.number_input(
        "Torque (Nm)",
        value=40.0
    )

    wear=st.number_input(
        "Tool Wear (min)",
        value=100.0
    )

    if st.button("Predict"):

        machine=encoder.transform([machine])[0]

        sample=pd.DataFrame([{

            "Type":machine,
            "Air temperature [K]":air,
            "Process temperature [K]":process,
            "Rotational speed [rpm]":speed,
            "Torque [Nm]":torque,
            "Tool wear [min]":wear

        }])

        probability=model.predict_proba(sample)[0][1]

        prediction=model.predict(sample)[0]

        st.metric(
            "Failure Probability",
            f"{probability*100:.2f}%"
        )

        if probability>0.6:
            st.error("🔴 High Risk")

        elif probability>0.3:
            st.warning("🟠 Medium Risk")

        else:
            st.success("🟢 Low Risk")

        if prediction==1:
            st.error("Machine Failure Expected")

        else:
            st.success("Machine is Healthy")

        fig=go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            number={"suffix":"%"},
            title={"text":"Failure Probability"},
            gauge={
                "axis":{"range":[0,100]}
            }
        ))

        st.plotly_chart(fig,use_container_width=True)

        # =====================================
# Explainability
# =====================================
elif page == "Explainability":

    st.title("🧠 Model Explainability")

    sample = X_test[features].sample(300, random_state=1)

    # Latest SHAP compatible
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    importance = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    st.subheader("Feature Importance")

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance Table")

    st.dataframe(
        importance_df,
        use_container_width=True
    )

    st.subheader("Dependence Plot")

    feature = st.selectbox(
        "Select Feature",
        features
    )

    index = features.index(feature)

    dependence = pd.DataFrame({
        "Feature Value": sample[feature],
        "SHAP Value": shap_values[:, index]
    })

    fig = px.scatter(
        dependence,
        x="Feature Value",
        y="SHAP Value",
        trendline="ols"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================
# Data Explorer
# =====================================
elif page == "Data Explorer":

    st.title("📁 Data Explorer")

    type_filter = st.multiselect(
        "Machine Type",
        data["Type"].unique(),
        default=data["Type"].unique()
    )

    failure = st.selectbox(
        "Machine Failure",
        [
            "All",
            "Failure",
            "No Failure"
        ]
    )

    df = data[data["Type"].isin(type_filter)]

    if failure == "Failure":
        df = df[df["Machine failure"] == 1]

    elif failure == "No Failure":
        df = df[df["Machine failure"] == 0]

    st.dataframe(
        df,
        use_container_width=True,
        height=450
    )

    st.write("Total Records :", len(df))

    st.subheader("Correlation Matrix")

    corr = data.select_dtypes(include=np.number).corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    csv = df.to_csv(index=False)

    st.download_button(
        "Download CSV",
        csv,
        "AI4I_Data.csv",
        "text/csv"
    )

