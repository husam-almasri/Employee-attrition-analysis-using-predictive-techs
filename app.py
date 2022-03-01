from flask import Flask, request, render_template
from flask_cors import cross_origin
import numpy as np
import joblib

app = Flask(__name__)
# model = pickle.load(open("saved_models/RandomForestClassifier.pkl", "rb"))
Attrition_clf=joblib.load("saved_models/RandomForestClassifier.pkl")
years_at_company_reg=joblib.load("saved_models/GradientBoostingRegressor.pkl")
# Attrition classification function with predict the years at the company if the attrition is yes
def Attrition_classification(
    Age,
    BusinessTravel,
    Department,
    DistanceFromHome,
    Education,
    EducationField,
    Gender,
    JobLevel,
    JobRole,
    MaritalStatus,
    NumCompaniesWorked,
    PercentSalaryHike,
    StockOptionLevel,
    TotalWorkingYears,
    TrainingTimesLastYear,
    YearsAtCompany,
    YearsSinceLastPromotion,
    absences_days,
    mean_actual_hours,
    JobInvolvement,
    PerformanceRating,
    EnvironmentSatisfaction,
    JobSatisfaction,
    WorkLifeBalance,
    MonthlyIncome
    ):
    attrs=np.zeros(25)
    attrs[0]=Age
    attrs[1]=BusinessTravel
    attrs[2]=Department
    attrs[3]=DistanceFromHome
    attrs[4]=Education
    attrs[5]=EducationField
    attrs[6]=Gender
    attrs[7]=JobLevel
    attrs[8]=JobRole
    attrs[9]=MaritalStatus
    attrs[10]=NumCompaniesWorked
    attrs[11]=PercentSalaryHike
    attrs[12]=StockOptionLevel
    attrs[13]=TotalWorkingYears
    attrs[14]=TrainingTimesLastYear
    attrs[15]=YearsAtCompany
    attrs[16]=YearsSinceLastPromotion
    attrs[17]=absences_days
    attrs[18]=mean_actual_hours
    attrs[19]=JobInvolvement
    attrs[20]=PerformanceRating
    attrs[21]=EnvironmentSatisfaction
    attrs[22]=JobSatisfaction
    attrs[23]=WorkLifeBalance
    attrs[24]=MonthlyIncome
#     Classification prediction for ATTRITION
    attrition=Attrition_clf.predict([attrs])[0]
#     Drop YearsAtCompany feature from the arry because it's the target
    attrs2=np.delete(attrs, obj=15)
#     Prediction for number of years at the company
    years_at_company=years_at_company_reg.predict([attrs2])[0]
    return attrition, years_at_company


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

@app.route("/predict", methods = ["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        #     Age of the employee (18-60)
        Age=request.form['Age']
        #     'Non-Travel','Travel_Frequently','Travel_Rarely'
        BusinessTravel=request.form['BusinessTravel']
        #     'Human Resources', 'Research & Development', 'Sales'
        Department=request.form['Department']
        #     Distance from home in kms
        DistanceFromHome=request.form['DistanceFromHome']
        #     'Below College','College','Bachelor','Master','Doctor'
        Education=request.form['Education']
        #     'Human Resources','Life Sciences','Marketing','Medical','Other','Technical Degree'
        EducationField=request.form['EducationField']
        #     'Female', 'Male'
        Gender=request.form['Gender']
        #     A scale of 1 to 5
        JobLevel=request.form['JobLevel']
        #     'Healthcare Representative','Human Resources','Laboratory Technician','Manager',
        #     'Manufacturing Director','Research Director','Research Scientist','Sales Executive',
        #     'Sales Representative'
        JobRole=request.form['JobRole']
        #     'Divorced', 'Married', 'Single'
        MaritalStatus=request.form['MaritalStatus']
        #     Total number of companies the employee has worked for
        NumCompaniesWorked=request.form['NumCompaniesWorked']
        #     Percent salary hike for last year
        PercentSalaryHike=request.form['PercentSalaryHike']
        #     Stock option level of the employee
        StockOptionLevel=request.form['StockOptionLevel']
        #     Total number of years the employee has worked so far
        TotalWorkingYears=request.form['TotalWorkingYears']
        #     Number of times training was conducted for this employee last year
        TrainingTimesLastYear=request.form['TrainingTimesLastYear']
        #     Total number of years spent at the company by the employee
        YearsAtCompany=request.form['YearsAtCompany']
        #     Number of years since last promotion
        YearsSinceLastPromotion=request.form['YearsSinceLastPromotion']
        #     Number of days that the employee didn't show up in 2015
        absences_days=request.form['absences_days']
        #     Mean of actual of work-hours in 2015
        mean_actual_hours=request.form['mean_actual_hours']
        #     'Low', 'Medium', 'High', 'Very High'
        JobInvolvement=request.form['JobInvolvement']
        #     'Low', 'Good', 'Excellent', 'Outstanding'
        PerformanceRating=request.form['PerformanceRating']
        #     'Low', 'Medium', 'High', 'Very High'
        EnvironmentSatisfaction=request.form['EnvironmentSatisfaction']
        #     'Low', 'Medium', 'High', 'Very High'
        JobSatisfaction=request.form['JobSatisfaction']
        #     'Bad', 'Good', 'Better', 'Best'
        WorkLifeBalance=request.form['WorkLifeBalance']
        #     Monthly income in rupees per month
        MonthlyIncome=request.form['MonthlyIncome']

        Name = request.form['Name']

        attrition, years_at_company = Attrition_classification(
        Age,
        BusinessTravel,
        Department,
        DistanceFromHome,
        Education,
        EducationField,
        Gender,
        JobLevel,
        JobRole,
        MaritalStatus,
        NumCompaniesWorked,
        PercentSalaryHike,
        StockOptionLevel,
        TotalWorkingYears,
        TrainingTimesLastYear,
        YearsAtCompany,
        YearsSinceLastPromotion,
        absences_days,
        mean_actual_hours,
        JobInvolvement,
        PerformanceRating,
        EnvironmentSatisfaction,
        JobSatisfaction,
        WorkLifeBalance,
        MonthlyIncome
        )
        if attrition == 1:
            years_at_company=round(years_at_company,2)
            return render_template('home.html',results=f"There is a high probability that {Name} will LEAVE the company in {years_at_company} years")
        return render_template('home.html',results=f"There is a high probability that {Name} will STAY at the company")

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
