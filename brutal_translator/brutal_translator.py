import string


class BrutalTranslator:
    def __init__(self, dictionary_path="english_polish.txt"):
        # Dictionary to hold our translations
        self.translation_dict = {}

        # Special case words that shouldn't appear as Polish translations
        self.english_words_to_check = ["from", "about", "to", "the", "a", "an"]

        # Load translations from txt file
        self.load_dictionary(dictionary_path)

        # Fix problematic translations
        self.fix_translations()

    def load_dictionary(self, dictionary_path):
        """Load translations from the txt file"""
        try:
            with open(dictionary_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

                for line in lines:
                    if '\t' in line:
                        # Split by tab character
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            english = parts[0].strip().lower()
                            polish = parts[1].strip()

                            # Store in dictionary
                            if english not in self.translation_dict:
                                self.translation_dict[english] = []

                            self.translation_dict[english].append(polish)

                print(f"Loaded {len(self.translation_dict)} unique English words")

                # Print some statistics
                total_translations = sum(len(translations) for translations in self.translation_dict.values())
                print(f"Total translations: {total_translations}")
                print(f"Average translations per word: {total_translations / len(self.translation_dict):.2f}")

        except FileNotFoundError:
            print(f"Error: Dictionary file '{dictionary_path}' not found")
        except Exception as e:
            print(f"Error loading dictionary: {e}")

    def fix_translations(self):
        """Fix problematic translations and add missing common words"""
        # Check for English words that appear as Polish translations and remove them
        for eng, pol_list in list(self.translation_dict.items()):
            fixed_list = []
            for pol in pol_list:
                if pol.lower() != eng.lower() and pol.lower() not in self.english_words_to_check:
                    fixed_list.append(pol)
                # else:
                #     print(f"Removed problematic translation: {eng} -> {pol}")

            # If we have valid translations left, update the list
            if fixed_list:
                self.translation_dict[eng] = fixed_list

        # Add translations for common words that might be missing
        common_translations = {
            "the": "",
            "a": "",
            "an": "",
            "to": "do",
            "about": "o",
            "of": "z",
            "in": "w",
            "on": "na",
            "at": "przy",
            "by": "przez",
            "is": "jest",
            "are": "są",
            "am": "jestem",
            "was": "był",
            "were": "były",
            "from": "z",
            "with": "z",
            "i": "ja",
            "my": "mój",
        }

        # Add these translations if they're not already present
        for eng, pol in common_translations.items():
            if eng not in self.translation_dict:
                self.translation_dict[eng] = [pol]
            elif pol and pol not in self.translation_dict[eng]:
                self.translation_dict[eng].append(pol)

    def translate_word(self, word):
        """Translate a single word preserving case and punctuation"""
        # Handle empty words
        if not word:
            return word

        # Extract punctuation
        prefix_punct = ""
        word_only = ""
        suffix_punct = ""

        # Extract leading punctuation
        i = 0
        while i < len(word) and word[i] in string.punctuation:
            prefix_punct += word[i]
            i += 1

        # Extract trailing punctuation
        j = len(word) - 1
        while j >= i and word[j] in string.punctuation:
            suffix_punct = word[j] + suffix_punct
            j -= 1

        # Extract the word itself
        if i <= j:
            word_only = word[i:j + 1]

        # Skip translation for empty words
        if not word_only:
            return word

        # Check for translation
        clean_word = word_only.lower()

        if clean_word in self.translation_dict and self.translation_dict[clean_word]:
            # Always pick the first translation for consistency
            translated = self.translation_dict[clean_word][0]

            # Skip empty translations for articles
            if not translated:
                return prefix_punct + suffix_punct

            # Preserve original capitalization
            if word_only[0].isupper():
                translated = translated[0].upper() + translated[1:] if translated else ""

            return prefix_punct + translated + suffix_punct
        else:
            # If no translation found, return the original word
            return word

    def translate(self, english_text):
        """Translate English text to Polish word by word"""
        if not english_text:
            return ""

        # Split text into words
        words = english_text.split()
        translated_words = []

        # Translate each word individually
        for word in words:
            translated = self.translate_word(word)
            # Don't add empty translations to the result
            if translated:
                translated_words.append(translated)

        # Join words back into text
        return " ".join(translated_words)


if __name__ == "__main__":
    # Path to dictionary file
    dictionary_path = "data/MUSEMultilingualEmbeddings.txt"

    # Create translator
    translator = BrutalTranslator(dictionary_path)

    # Test sentences
    test_sentences = [
        "This page also has new articles that were not for you.",
        "First, they had one article which was talking about his utc page.",
        "You are from the new page but they were not.",
        "Who was the first to talk about this article?",
        "They also had a new page and were not the first one."
    ]

    print("\n=== BRUTAL ENGLISH TO POLISH TRANSLATOR ===\n")
    for sentence in test_sentences:
        print(f"English: {sentence}")
        print(f"Polish:  {translator.translate(sentence)}")
        print()

    print("Enter your own sentences (type 'exit' to quit):")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            break
        print(f"Polish: {translator.translate(user_input)}")