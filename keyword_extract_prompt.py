KEYWORD_EXTRACTION_PROMPT = """
Extract the keywords from the given paragraph.
Extract proper nouns as the first priority and nouns as the second priority to select about 4 words that can describe this paragraph.
Just return the key word in CSV format. Remember, maximum is 4 words.

Paragraph:
"""
