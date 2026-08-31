from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)


# =====================================================
# LOAD MODEL
# =====================================================

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# =====================================================
# LOAD SCALER
# =====================================================

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# =====================================================
# LOAD ONE HOT ENCODER
# =====================================================

with open("ohe.pkl", "rb") as file:
    ohe = pickle.load(file)


# =====================================================
# LOAD LABEL ENCODER
# =====================================================

with open("education_encoder.pkl", "rb") as file:
    education_le = pickle.load(file)


# =====================================================
# LOAD FEATURE COLUMNS
# =====================================================

with open("feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)


# =====================================================
# HOME PAGE
# =====================================================

@app.route("/")
def home():
    return render_template("index.html")


# =====================================================
# PREDICTION
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # =================================================
        # NUMERICAL INPUTS
        # =================================================

        applicant_income = float(
            request.form["Applicant_Income"]
        )

        coapplicant_income = float(
            request.form["Coapplicant_Income"]
        )

        age = float(
            request.form["Age"]
        )

        dependents = float(
            request.form["Dependents"]
        )

        credit_score = float(
            request.form["Credit_Score"]
        )

        existing_loans = float(
            request.form["Existing_Loans"]
        )

        dti_ratio = float(
            request.form["DTI_Ratio"]
        )

        savings = float(
            request.form["Savings"]
        )

        collateral_value = float(
            request.form["Collateral_Value"]
        )

        loan_amount = float(
            request.form["Loan_Amount"]
        )

        loan_term = float(
            request.form["Loan_Term"]
        )


        # =================================================
        # CATEGORICAL INPUTS
        # =================================================

        education_level = request.form["Education_Level"]

        employment_status = request.form["Employment_Status"]

        marital_status = request.form["Marital_Status"]

        loan_purpose = request.form["Loan_Purpose"]

        property_area = request.form["Property_Area"]

        gender = request.form["Gender"]

        employer_category = request.form["Employer_Category"]


        # =================================================
        # CHECK CATEGORIES
        # =================================================

        categorical_data = {
            "Employment_Status": employment_status,
            "Marital_Status": marital_status,
            "Loan_Purpose": loan_purpose,
            "Property_Area": property_area,
            "Gender": gender,
            "Employer_Category": employer_category
        }


        for column, value in categorical_data.items():

            column_index = list(ohe.feature_names_in_).index(column)

            allowed_values = ohe.categories_[column_index]

            if value not in allowed_values:

                return render_template(
                    "index.html",
                    prediction_text=f"Invalid value for {column}: {value}. "
                                     f"Allowed values: {', '.join(allowed_values)}"
                )


        # =================================================
        # EDUCATION LABEL ENCODING
        # =================================================

        try:

            education_level_encoded = education_le.transform(
            [education_level]
            )[0]

        except ValueError:

            return render_template(
                "index.html",
            prediction_text=(
                f"Invalid Education_Level: {education_level}. "
                f"Allowed values: {list(education_le.classes_)}"
            )
        )


        # =================================================
        # CREATE ORIGINAL DATA
        # =================================================

        data = pd.DataFrame([{

            "Applicant_Income": applicant_income,

            "Coapplicant_Income": coapplicant_income,

            "Age": age,

            "Dependents": dependents,

            "Existing_Loans": existing_loans,

            "Savings": savings,

            "Collateral_Value": collateral_value,

            "Loan_Amount": loan_amount,

            "Loan_Term": loan_term,

            "Education_Level": education_level_encoded,

            "DTI_Ratio_sq": dti_ratio ** 2,

            "Credit_Score_sq": credit_score ** 2,

            "Employment_Status": employment_status,

            "Marital_Status": marital_status,

            "Loan_Purpose": loan_purpose,

            "Property_Area": property_area,

            "Gender": gender,

            "Employer_Category": employer_category

        }])


        # =================================================
        # CATEGORICAL COLUMNS
        # =================================================

        categorical_cols = [
            "Employment_Status",
            "Marital_Status",
            "Loan_Purpose",
            "Property_Area",
            "Gender",
            "Employer_Category"
        ]


        # =================================================
        # ONE HOT ENCODING
        # =================================================

        encoded = ohe.transform(
            data[categorical_cols]
        )


        encoded_df = pd.DataFrame(
            encoded,
            columns=ohe.get_feature_names_out(
                categorical_cols
            ),
            index=data.index
        )


        # =================================================
        # REMOVE ORIGINAL CATEGORICAL COLUMNS
        # =================================================

        data = data.drop(
            columns=categorical_cols
        )


        # =================================================
        # COMBINE DATA
        # =================================================

        final_data = pd.concat(
            [
                data,
                encoded_df
            ],
            axis=1
        )


        # =================================================
        # EXACT SAME FEATURE ORDER AS TRAINING
        # =================================================

        final_data = final_data.reindex(
            columns=feature_columns,
            fill_value=0
        )


        # =================================================
        # DEBUG
        # =================================================

        print("\n================================")
        print("FINAL FEATURES")
        print("================================")

        print(final_data.columns.tolist())

        print("\n================================")
        print("FINAL DATA")
        print("================================")

        print(final_data)


        # =================================================
        # SCALE
        # =================================================

        final_scaled = scaler.transform(
            final_data
        )


        # =================================================
        # PREDICTION
        # =================================================

        prediction = model.predict(
            final_scaled
        )[0]


        # =================================================
        # PROBABILITY
        # =================================================

        probability = model.predict_proba(
            final_scaled
        )[0][1]


        probability_percent = round(
            probability * 100,
            2
        )


        # =================================================
        # RESULT
        # =================================================

        if prediction == 1:

            output = (
                f"Loan Approved "
                f"(Probability: {probability_percent}%)"
            )

        else:

            output = (
                f"Loan Not Approved "
                f"(Probability: {probability_percent}%)"
            )


        return render_template(
            "index.html",
            prediction_text=output
        )


    except Exception as e:

        print("\nERROR:")
        print(e)

        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


# =====================================================
# RUN FLASK
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)