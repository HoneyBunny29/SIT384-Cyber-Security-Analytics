import pandas as pd  # Importing the pandas library for data manipulation

# Read the CSV file
df = pd.read_csv("C:/Users/corer/OneDrive/Desktop/2nd year/C.S analysitic/Week 03/3.2C/result_withoutTotal.csv")

# Calculate the Total score using the given formula
df['Total'] = 0.05 * (df['Ass1'] + df['Ass3']) + 0.15 * (df['Ass2'] + df['Ass4']) + 0.5 * df['Exam']

# Calculate the Final score by rounding the Total score to the nearest integer
df['Final'] = df['Total'].round()

# Apply the hurdle rule
# Students must have a Total >= 50 and Exam >= 48 to pass the unit
# If a student fails the hurdle (Exam < 48), the max Final is 44
df.loc[df['Exam'] < 48, 'Final'] = df.loc[df['Exam'] < 48, 'Final'].apply(lambda x: min(x, 44))

# Assign grades based on the Final score
def assign_grade(final_score):
    if final_score <= 49.45:
        return 'N'
    elif final_score <= 59.45:
        return 'P'
    elif final_score <= 69.45:
        return 'C'
    elif final_score <= 79.45:
        return 'D'
    else:
        return 'HD'

df['Grade'] = df['Final'].apply(assign_grade)

# Save the result data with the new columns to a new file
df.to_csv("C:/Users/corer/OneDrive/Desktop/2nd year/C.S analysitic/Week 03/3.2C/result_updated.csv", index=False)

# Save the records of students with exam score < 48 to a new file
failed_hurdle = df[df['Exam'] < 48]
failed_hurdle.to_csv("C:/Users/corer/OneDrive/Desktop/2nd year/C.S analysitic/Week 03/3.2C/failedhurdle.csv", index=False)

# Display the updated result data
print("Updated result data with Total, Final, and Grade columns:")
print(df.to_string(index=False))  # Displaying without the index for a cleaner look

# Display the students with exam score < 48
print("\nStudents who failed the hurdle (Exam < 48):")
print(failed_hurdle.to_string(index=False))  # Displaying without the index for a cleaner look

# Display the students with exam score > 100
print("\nStudents with exam score > 100:")
print(df[df['Exam'] > 100].to_string(index=False))  # Displaying without the index for a cleaner look