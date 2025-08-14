import easyocr

reader = easyocr.Reader(['en'])  
results = reader.readtext(r"C:\\image_folder\\new_tax_2.PNG")
for bbox, text, prob in results:
    print(f"{text} ({prob})")