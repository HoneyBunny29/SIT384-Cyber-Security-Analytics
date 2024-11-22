import re

def validate_email(email):
    # Regular expression pattern for a valid email address
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    
    if re.match(pattern, email):
        return True
    else:
        return False

def extract_email_parts(email):
    username, host = email.split('@')
    return username, host

while True:
    # Get email input from the user
    email = input("Please input your email address: ")
    
    # Validate the email
    if validate_email(email):
        username, host = extract_email_parts(email)
        print(f"Email: {email}")
        print(f"Username: {username}")
        print(f"Host: {host}")
        break
    else:
        print("Not a valid email. Please try again.")