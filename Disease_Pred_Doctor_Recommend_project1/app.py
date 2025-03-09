from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import json

# Load pickle file
with open("disease_model.pkl", "rb") as file:
    data = pickle.load(file)

app = Flask(__name__)

# Extract loaded data
model = data["model"]
symptoms_list = data["symptoms_list"]
unique_location = data["unique_location"]
unique_disease = data["unique_disease"]
df_descr = data["df_descr"]
df_advice = data["df_advice"]
df_dr = data["df_dr"]
df_dis_sym = data["df_dis_sym"]

def generate_one_hot_vector(matched_symptoms):
    # Ensure the feature names match the model's expected input
    one_hot_vector = {symptom: 0 for symptom in model.feature_names_in_}  # Use model's feature names
    for symptom in matched_symptoms:
        if symptom in one_hot_vector:
            one_hot_vector[symptom] = 1
    return pd.DataFrame([one_hot_vector])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_disease", methods=["GET", "POST"])
def predict_disease():
    if request.method == "GET":
        # Render the prediction page with the list of symptoms
        return render_template("predict.html", symptoms_list=symptoms_list)

    elif request.method == "POST":
        try:
            # Extract symptoms from the JSON payload
            user_input_string = request.json.get("symptoms", [])
            if not user_input_string:
                return jsonify({"error": "No symptoms provided"}), 400

            # Process the symptoms for prediction
            user_inputs = [symptom.strip().replace(" ", "_") for symptom in user_input_string]

            # Generate a one-hot encoded vector for the symptoms
            one_hot_vector = generate_one_hot_vector(user_inputs)
            if one_hot_vector is None or one_hot_vector.empty:
                return jsonify({"error": "Failed to generate feature vector for symptoms"}), 400
            
            # Predict probabilities using the model
            probabilities = model.predict_proba(one_hot_vector)[0]

            # Map diseases to their respective probabilities
            disease_probabilities = {disease: prob for disease, prob in zip(model.classes_, probabilities)}

            # Get the top 5 diseases sorted by probability
            top_diseases = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]

            # Prepare the response data
            response = {
                "top_diseases": [{"disease": disease, "probability": round(prob * 100, 2)} for disease, prob in top_diseases]
            }
            return jsonify(response)

        except Exception as e:
            # Debug: Log the error for troubleshooting
            print(f"Error: {e}")
            return jsonify({"error": "Internal server error"}), 500


@app.route("/get_precaution_description", methods=["POST"])
def get_precaution_description():
    try:
        # Get top 2 diseases from the request
        top_diseases = request.json.get("top_diseases", [])

        # Filter the advice and description datasets
        filtered_advice = df_advice[df_advice['Disease'].isin(top_diseases)]
        filtered_descriptions = df_descr[df_descr['Disease'].isin(top_diseases)]

        # Merge datasets on 'Disease'
        merged_data = pd.merge(filtered_advice, filtered_descriptions, on='Disease')

        # Combine precautions into a single list per disease
        merged_data['Combined_Precautions'] = merged_data.apply(
            lambda row: [row[col] for col in df_advice.columns[1:5] if pd.notnull(row[col])],
            axis=1
        )

        # Prepare the response
        results = []
        for _, row in merged_data.iterrows():
            results.append({
                "disease": row["Disease"],
                "description": row["Description"],
                "precautions": row["Combined_Precautions"]
            })
        return jsonify(results)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/recommend", methods=["GET", "POST"])
def recommended_doctors():
    if request.method == "POST":
        try:
            # Extract the disease and location from the POST request
            data = request.get_json()
            disease = data.get("disease", "")
            location = data.get("location", "")

            # Validate inputs
            if not disease or not location:
                return jsonify({"error": "Disease or location not provided"}), 400

            # Fetch specialization for the disease
            specialization = df_dis_sym.loc[df_dis_sym['Disease'] == disease, 'Specialization'].values
            if len(specialization) == 0:
                return jsonify({"error": "No specialization found for this disease"}), 404

            specialization = specialization[0]

            # Filter doctors based on specialization and location
            filtered_doctors = df_dr[(df_dr['Specialization'] == specialization) & (df_dr['Location'] == location)]
            
            # If no doctors are found, check for "General Medicine" specialization
            if filtered_doctors.empty:
                filtered_doctors = df_dr[(df_dr['Specialization'] == "General Medicine") & (df_dr['Location'] == location)]
            
            if filtered_doctors.empty:
                return jsonify({"error": "No doctors available for the given specialization and location."}), 404
            
            # Calculate scores for doctors
            filtered_doctors['score'] = (
                filtered_doctors['normalized weighted average'] * 0.8 +
                filtered_doctors['normalized experience'] * 0.2
            )

            # Convert scores to percentages
            filtered_doctors['score_percentage'] = (filtered_doctors['score'] * 100).round(2)

            # Sort by score and prepare doctor data 
            recommended_doctors = filtered_doctors.sort_values(by='score', ascending=False)
        
            doctor_list = recommended_doctors[["Doctor ID", "Doctor Name", "Specialization", "Patient Rating", "Experience (Years)", "Consultation Fee ($)", "Availability","Insurance Accepted", "score_percentage"]].rename(columns={"score_percentage": "Score"}).to_dict(orient="records")

            return jsonify(doctor_list)
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({"error": "Internal server error"}), 500
    else:
        # Render template and pass disease if available via query parameters
        disease = request.args.get("disease", "")
        return render_template("recommend.html", disease=disease, diseases=unique_disease, locations=unique_location)


if __name__ == "__main__":
    app.run(debug=True)
   