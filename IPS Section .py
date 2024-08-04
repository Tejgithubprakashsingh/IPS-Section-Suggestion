import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util
from tkinter import Tk, Label, Entry, Text, Button, END, font

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    preprocessed_text = ' '.join(words)
    return preprocessed_text

# Load preprocessed dataset
new_ds = pickle.load(open("C:/Users/hp/OneDrive/Desktop/python1/preprocess_data.pkl", 'rb'))

# Load SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Suggest sections function
def suggest_sections(complaint, dataset, min_suggestions=5):
    preprocessed_complaint = preprocess_text(complaint)
    complaint_embedding = model.encode(preprocessed_complaint)
    section_embedding = model.encode(dataset['Combo'].tolist())
    similarities = util.pytorch_cos_sim(complaint_embedding, section_embedding)[0]
    similarity_threshold = 0.2
    relevant_indices = []
    while len(relevant_indices) < min_suggestions and similarity_threshold > 0:
        relevant_indices = [i for i, sim in enumerate(similarities) if sim > similarity_threshold]
        similarity_threshold -= 0.05
    sorted_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)
    suggestions = dataset.iloc[sorted_indices][['Description', 'Offense', 'Punishment', 'Cognizable', 'Bailable', 'Court', 'Combo']].to_dict(orient='records')
    return suggestions

# GUI application
def get_suggestion():
    complaint = complaint_entry.get()
    suggestions = suggest_sections(complaint, new_ds)
    output_text.delete(1.0, END)
    if suggestions:
        output_text.insert(END, "Suggested IPS Sections are:\n", 'title')
        for suggestion in suggestions:
            output_text.insert(END, "Description:\n", 'heading')
            output_text.insert(END, f"{suggestion['Description']}\n\n", 'description')
            output_text.insert(END, "Offense:\n", 'heading')
            output_text.insert(END, f"{suggestion['Offense']}\n\n", 'offense')
            output_text.insert(END, "Punishment:\n", 'heading')
            output_text.insert(END, f"{suggestion['Punishment']}\n\n", 'punishment')
            output_text.insert(END, "----------------------------------------\n", 'separator')

# Create GUI window
root = Tk()
root.title("IPS Section Suggestion")

# Set fonts
title_font = font.Font(root, family='Helvetica', size=14, weight='bold')
heading_font = font.Font(root, family='Helvetica', size=12, weight='bold')
description_font = font.Font(root, family='Helvetica', size=12)
offense_font = font.Font(root, family='Helvetica', size=12, slant='italic')
punishment_font = font.Font(root, family='Helvetica', size=12, underline=True)
separator_font = font.Font(root, family='Helvetica', size=10)

complaint_label = Label(root, text="Enter crime description")
complaint_label.pack()

complaint_entry = Entry(root, width=100)
complaint_entry.pack()

suggest_button = Button(root, text="Get Suggestion", command=get_suggestion)
suggest_button.pack()

output_text = Text(root, width=100, height=20, wrap='word')
output_text.pack()

# Add tags for styling
output_text.tag_configure('title', font=title_font, foreground='blue')
output_text.tag_configure('heading', font=heading_font, foreground='darkred')
output_text.tag_configure('description', font=description_font, foreground='black')
output_text.tag_configure('offense', font=offense_font, foreground='purple')
output_text.tag_configure('punishment', font=punishment_font, foreground='green')
output_text.tag_configure('separator', font=separator_font, foreground='gray')

root.mainloop()
