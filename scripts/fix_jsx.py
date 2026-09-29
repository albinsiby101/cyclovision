path = r"C:\Users\albin\Documents\cyclo\frontend\src\App.jsx"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Replace raw >= in JSX with &ge;
text = text.replace(">=30kt", "&ge; 30kt")
text = text.replace(">= 120 kt", "&ge; 120 kt")

with open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("Escaped JSX entities in App.jsx")