while True:
    try:
        # Get an integer from the user
        n = int(input("Please input a positive integer: "))
        
        # Check if the input is a positive integer
        if n <= 0:
            print ("Integer should be greater than 0")
        else:
            break
    except ValueError:
        print("Sorry, invaild input. Please enter an integer")
        
for i in range(n):
    print("*" * n)
