import re


def remove_extra_chars(prompt):
    # Remove leading tabs and spaces from lines
    processed_prompt = re.sub(r"^[ \t]+", "", prompt, flags=re.MULTILINE)

    # Replace occurrences of more than two consecutive new lines with exactly two new lines
    processed_prompt = re.sub(r"\n{3,}", "\n\n", processed_prompt)

    return processed_prompt
