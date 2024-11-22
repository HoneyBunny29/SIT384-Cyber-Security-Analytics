def recursive_factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * recursive_factorial(n - 1)

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

# Call the recursive factorial function and display the result
print(f"Factorial of {num}: {recursive_factorial(num)}")