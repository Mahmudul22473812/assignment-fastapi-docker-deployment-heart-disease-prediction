from pydantic import BaseModel, ConfigDict, Field


class HeartFeatures(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    age: int = Field(gt=0, le=120, description="Age in years")
    sex: int = Field(ge=0, le=1, description="0 female, 1 male")
    cp: int = Field(ge=0, le=3, description="Chest pain type in the Kaggle encoding")
    trestbps: int = Field(gt=0, le=300, description="Resting blood pressure in mm Hg")
    chol: int = Field(gt=0, le=1000, description="Serum cholesterol in mg/dL")
    fbs: int = Field(ge=0, le=1, description="1 if fasting blood sugar is above 120 mg/dL")
    restecg: int = Field(ge=0, le=2, description="Resting ECG result in the Kaggle encoding")
    thalach: int = Field(gt=0, le=300, description="Maximum heart rate achieved")
    exang: int = Field(ge=0, le=1, description="1 if exercise induced angina")
    oldpeak: float = Field(ge=0, le=20, description="Exercise-induced ST depression relative to rest")
    slope: int = Field(ge=0, le=2, description="Peak exercise ST slope in the Kaggle encoding")
    ca: int = Field(ge=0, le=4, description="Major vessels seen by fluoroscopy. 4 encodes missing.")
    thal: int = Field(ge=0, le=3, description="Thallium test result. 0 encodes missing.")


FEATURE_NAMES = tuple(HeartFeatures.model_fields)


class HeartPrediction(BaseModel):
    heart_disease: bool
