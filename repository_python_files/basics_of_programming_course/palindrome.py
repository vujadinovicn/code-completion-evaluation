def palindrome(text):
    text = text.lower()
    is_palindrome = True
    for i in range(len(text) // 2):
        if text[i] != text[-1 - i]:
            is_palindrome = False
    return is_palindrome