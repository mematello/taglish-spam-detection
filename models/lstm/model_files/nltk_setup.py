# NLTK Setup Script - Run this ONCE before using the spam detection code
import nltk

print("Downloading required NLTK data...")

# Download all required NLTK packages
packages = ['punkt', 'punkt_tab', 'stopwords']

for package in packages:
    try:
        print(f"Downloading {package}...")
        nltk.download(package, quiet=False)
        print(f"✓ {package} downloaded successfully")
    except Exception as e:
        print(f"Error downloading {package}: {e}")

print("\nNLTK setup complete! You can now run the spam detection code.")

# Test the downloads
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    # Test tokenization
    test_text = "This is a test sentence."
    tokens = word_tokenize(test_text)
    print(f"✓ Tokenization test: {tokens}")
    
    # Test stopwords
    stop_words = stopwords.words('english')
    print(f"✓ Stopwords loaded: {len(stop_words)} words")
    
    print("\nAll NLTK components are working correctly!")
    
except Exception as e:
    print(f"Error testing NLTK: {e}")