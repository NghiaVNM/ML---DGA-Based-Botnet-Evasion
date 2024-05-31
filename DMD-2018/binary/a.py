def create_char_to_int_mapping(s):
    unique_chars = sorted(set(s))  # Get unique characters and sort them
    char_to_int_mapping = {char: idx for idx, char in enumerate(unique_chars)}
    return char_to_int_mapping

# Example usage
input_string = "hello world"
mapping = create_char_to_int_mapping(input_string)
print(mapping)