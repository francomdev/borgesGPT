import os
from PyPDF2 import PdfReader

dataset = "" 

with open("dataset.txt", 'w') as f:
    for filename in os.listdir(os.getcwd()):
        if filename[-3:] == 'pdf':
            reader = PdfReader(filename)
            number_of_pages = len(reader.pages)
            page = reader.pages[9]
            text = page.extract_text()
            for i in range(number_of_pages):
                text = reader.pages[i].extract_text()
                f.write(text + '\n')

