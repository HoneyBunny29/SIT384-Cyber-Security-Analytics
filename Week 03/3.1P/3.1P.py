import pandas as pd

# Load the CSV file, selecting only the required columns
df = pd.read_csv('result_withoutTotal.csv', usecols=['ID','Ass1', 'Ass2', 'Ass3', 'Ass4', 'Exam'])

# Print the average of ass1, ass2, ass3, ass4 and exam
print("Average Scores:")
print("Ass1:", df['Ass1'].mean())
print("Ass2:", df['Ass2'].mean())
print("Ass3:", df['Ass3'].mean())
print("Ass4:", df['Ass4'].mean())
print("Exam:", df['Exam'].mean())

# Print the minimum of ass1, ass2, ass3, ass4 and exam
print("\nMinimum Scores:")
print("Ass1:", df['Ass1'].min())
print("Ass2:", df['Ass2'].min())
print("Ass3:", df['Ass3'].min())
print("Ass4:", df['Ass4'].min())
print("Exam:", df['Exam'].min())

# Print the maximum of ass1, ass2, ass3, ass4 and exam
print("\nMaximum Scores:")
print("Ass1:", df['Ass1'].max())
print("Ass2:", df['Ass2'].max())
print("Ass3:", df['Ass3'].max())
print("Ass4:", df['Ass4'].max())
print("Exam:", df['Exam'].max())

# Select students with the highest ass1, ass2, ass3, ass4 and exam and print their information
highest_ass1 = df[df['Ass1'] == df['Ass1'].max()]
highest_ass2 = df[df['Ass2'] == df['Ass2'].max()]
highest_ass3 = df[df['Ass3'] == df['Ass3'].max()]
highest_ass4 = df[df['Ass4'] == df['Ass4'].max()]
highest_exam = df[df['Exam'] == df['Exam'].max()]

print("\nStudent with highest Ass1:")
print(highest_ass1)

print("\nStudent with highest Ass2:")
print(highest_ass2)

print("\nStudent with highest Ass3:")
print(highest_ass3)

print("\nStudent with highest Ass4:")
print(highest_ass4)

print("\nStudent with highest Exam:")
print(highest_exam)
