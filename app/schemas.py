from pydantic import BaseModel, Field, AliasChoices

class StudentFeatures(BaseModel):
    Marital_status: int = Field(..., validation_alias=AliasChoices('Marital status', 'Marital_status'))
    Application_mode: int = Field(..., validation_alias=AliasChoices('Application mode', 'Application_mode'))
    Application_order: int = Field(..., validation_alias=AliasChoices('Application order', 'Application_order'))
    Course: int = Field(..., validation_alias=AliasChoices('Course'))
    Daytime_evening_attendance: int = Field(..., validation_alias=AliasChoices('Daytime/evening attendance', 'Daytime_evening_attendance'))
    Previous_qualification: int = Field(..., validation_alias=AliasChoices('Previous qualification', 'Previous_qualification'))
    Previous_qualification_grade: float = Field(..., validation_alias=AliasChoices('Previous qualification (grade)', 'Previous_qualification_grade'))
    Nacionality: int = Field(..., validation_alias=AliasChoices('Nacionality'))
    Mother_s_qualification: int = Field(..., validation_alias=AliasChoices("Mother's qualification", 'Mother_s_qualification'))
    Father_s_qualification: int = Field(..., validation_alias=AliasChoices("Father's qualification", 'Father_s_qualification'))
    Mother_s_occupation: int = Field(..., validation_alias=AliasChoices("Mother's occupation", 'Mother_s_occupation'))
    Father_s_occupation: int = Field(..., validation_alias=AliasChoices("Father's occupation", 'Father_s_occupation'))
    Admission_grade: float = Field(..., validation_alias=AliasChoices('Admission grade', 'Admission_grade'))
    Displaced: int = Field(..., validation_alias=AliasChoices('Displaced'))
    Educational_special_needs: int = Field(..., validation_alias=AliasChoices('Educational special needs', 'Educational_special_needs'))
    Debtor: int = Field(..., validation_alias=AliasChoices('Debtor'))
    Tuition_fees_up_to_date: int = Field(..., validation_alias=AliasChoices('Tuition fees up to date', 'Tuition_fees_up_to_date'))
    Gender: int = Field(..., validation_alias=AliasChoices('Gender'))
    Scholarship_holder: int = Field(..., validation_alias=AliasChoices('Scholarship holder', 'Scholarship_holder'))
    Age_at_enrollment: int = Field(..., validation_alias=AliasChoices('Age at enrollment', 'Age_at_enrollment'))
    International: int = Field(..., validation_alias=AliasChoices('International'))

    Curricular_units_1st_sem_credited: int = Field(..., validation_alias=AliasChoices('Curricular units 1st sem (credited)', 'Curricular_units_1st_sem_credited'))
    Curricular_units_1st_sem_enrolled: int = Field(..., validation_alias=AliasChoices('Curricular units 1st sem (enrolled)', 'Curricular_units_1st_sem_enrolled'))
    Curricular_units_1st_sem_evaluations: int = Field(..., validation_alias=AliasChoices('Curricular units 1st sem (evaluations)', 'Curricular_units_1st_sem_evaluations'))
    Curricular_units_1st_sem_approved: int = Field(..., validation_alias=AliasChoices('Curricular units 1st sem (approved)', 'Curricular_units_1st_sem_approved'))
    Curricular_units_1st_sem_grade: float = Field(..., validation_alias=AliasChoices('Curricular units 1st sem (grade)', 'Curricular_units_1st_sem_grade'))
    Curricular_units_1st_sem_without_evaluations: int = Field(..., validation_alias=AliasChoices('Curricular units 1st sem (without evaluations)', 'Curricular_units_1st_sem_without_evaluations'))

    Curricular_units_2nd_sem_credited: int = Field(..., validation_alias=AliasChoices('Curricular units 2nd sem (credited)', 'Curricular_units_2nd_sem_credited'))
    Curricular_units_2nd_sem_enrolled: int = Field(..., validation_alias=AliasChoices('Curricular units 2nd sem (enrolled)', 'Curricular_units_2nd_sem_enrolled'))
    Curricular_units_2nd_sem_evaluations: int = Field(..., validation_alias=AliasChoices('Curricular units 2nd sem (evaluations)', 'Curricular_units_2nd_sem_evaluations'))
    Curricular_units_2nd_sem_approved: int = Field(..., validation_alias=AliasChoices('Curricular units 2nd sem (approved)', 'Curricular_units_2nd_sem_approved'))
    Curricular_units_2nd_sem_grade: float = Field(..., validation_alias=AliasChoices('Curricular units 2nd sem (grade)', 'Curricular_units_2nd_sem_grade'))
    Curricular_units_2nd_sem_without_evaluations: int = Field(..., validation_alias=AliasChoices('Curricular units 2nd sem (without evaluations)', 'Curricular_units_2nd_sem_without_evaluations'))

    Unemployment_rate: float = Field(..., validation_alias=AliasChoices('Unemployment rate', 'Unemployment_rate'))
    Inflation_rate: float = Field(..., validation_alias=AliasChoices('Inflation rate', 'Inflation_rate'))
    GDP: float = Field(..., validation_alias=AliasChoices('GDP'))

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    prediction: str
    probability: dict