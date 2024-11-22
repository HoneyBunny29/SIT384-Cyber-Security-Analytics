def factorial(n):
    result = 1
    
    # Calculate factorial
    for i in range(1, n + 1):
        result *= i
    
    return result

while True:
    try:
        # Get a nonnegative integer from the user
        num = int(input("Please input a nonnegative integer: "))
        
        # Check if the number is nonnegative
        if num < 0:
            print("Please enter a nonnegative integer")
        else:
            break
    except ValueError:
        print("Please enter a valid integer")

# Call the factorial function and display the result
print(f"Factorial of {num}: {factorial(num)}")