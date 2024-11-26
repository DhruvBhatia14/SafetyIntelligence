import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import pickle

class InjuryPredictionModel:
    def __init__(self, file_path='FinalDataset29oct.csv'):
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)
        # self._prepare_data()
        self.best_models = {}
        self.best_model_scores = {}
        # Define preprocessing
        categorical_features = ['Task Assigned', 'Environmental Factor']
        numerical_features = ['Temp', 'Dps', 'FeelsLike', 'Heatind', 'Wchills', 'Precips']
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        self.label_encoder = LabelEncoder()


    def train_models(self):
        # Define body part mapping
        body_part_mapping = {
            'Head': 'Head/Neck', 'Face': 'Head/Neck', 'Eyes': 'Head/Neck', 'Ears': 'Head/Neck', 'Nose': 'Head/Neck',
            'Mouth': 'Head/Neck', 'Throat': 'Head/Neck', 'Neck': 'Head/Neck', 'Chest': 'Torso/Back', 'Abdomen': 'Torso/Back',
            'Pelvis': 'Torso/Back', 'Spine': 'Torso/Back', 'Back': 'Torso/Back', 'Shoulder': 'Upper Extremities (Arms/Hands)',
            'Arm': 'Upper Extremities (Arms/Hands)', 'Elbow': 'Upper Extremities (Arms/Hands)', 'Wrist': 'Upper Extremities (Arms/Hands)',
            'Hand': 'Upper Extremities (Arms/Hands)', 'Fingers': 'Upper Extremities (Arms/Hands)', 'Hip': 'Lower Extremities (Legs/Feet)',
            'Thigh': 'Lower Extremities (Legs/Feet)', 'Knee': 'Lower Extremities (Legs/Feet)', 'Leg': 'Lower Extremities (Legs/Feet)',
            'Ankle': 'Lower Extremities (Legs/Feet)', 'Feet': 'Lower Extremities (Legs/Feet)', 'Toes': 'Lower Extremities (Legs/Feet)',
            'Heart': 'Internal Organs', 'Lungs': 'Internal Organs', 'Stomach': 'Internal Organs', 'Intestines': 'Internal Organs',
            'Kidney': 'Internal Organs', 'Liver': 'Internal Organs', 'Multiple': 'Multiple Body Parts'
        }
        
        # Apply mapping
        self.data['Part of Body'] = self.data['Part of Body'].map(body_part_mapping)
        
        # Select required features and drop rows with missing values
        required_features = ['Part of Body', 'Task Assigned', 'Temp', 'Dps', 'FeelsLike', 'Heatind', 'Wchills', 'Precips', 'Environmental Factor']
        filtered_data = self.data[required_features].dropna()
        
        # Split features and target
        X = filtered_data.drop(columns='Part of Body')
        y = filtered_data['Part of Body']
        
        # Encode target variable
        # label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(y)
        # self.label_encoder = self.label_encoder  # Save encoder for later use

        # Train-validation-test split
        X_train, X_temp, y_train, y_temp = train_test_split(X, self.y_encoded, test_size=0.4, random_state=42, stratify=self.y_encoded)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        
        # Fit preprocessing on training data and transform all sets
        self.X_train_prepared = self.preprocessor.fit_transform(X_train)
        self.X_val_prepared = self.preprocessor.transform(self.X_val)
        self.X_test_prepared = self.preprocessor.transform(self.X_test)


        # Define models and parameter grids for GridSearchCV
        models = {
            "Decision Tree": (DecisionTreeClassifier(), {
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10]
            }),
            "Random Forest": (RandomForestClassifier(random_state=42), {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            }),
            "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            }),
            "Support Vector Machine": (SVC(probability=True), {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"]
            })
        }
        
        # Train each model with GridSearchCV
        for model_name, (model, param_grid) in models.items():
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
            # grid_search.fit(self.X_train_prepared, self.y_encoded)
            grid_search.fit(self.X_train_prepared, y_train)
            
            # Store the best model and its score
            self.best_models[model_name] = grid_search.best_estimator_
            self.best_model_scores[model_name] = grid_search.best_score_

    def get_best_models(self):
        return self.best_models, self.best_model_scores
    
    def svmres(self, new_data):
        # Ensure the data types match expected types for each column
        categorical_features = ['Task Assigned', 'Environmental Factor']
        numerical_features = ['Temp', 'Dps', 'FeelsLike', 'Heatind', 'Wchills', 'Precips']
        
        # Convert categorical columns to strings
        new_data[categorical_features] = new_data[categorical_features].astype(str)
        
        # Convert numerical columns to float
        new_data[numerical_features] = new_data[numerical_features].astype(float)
        
        # Now transform data
        new_data_prepared = self.preprocessor.transform(new_data)
        
        # Predict with the SVM model
        best_models, _ = self.get_best_models()
        svm_predictions = best_models["Support Vector Machine"].predict(new_data_prepared)
        predicted_body_part_svm = self.label_encoder.inverse_transform(svm_predictions)
        return predicted_body_part_svm

model = InjuryPredictionModel()
model.train_models()

# Pickle the model
with open('injury_prediction_model1.pkl', 'wb') as f:
    pickle.dump(model, f)


