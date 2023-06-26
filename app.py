import streamlit as st
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

categories = [
       'Bags, suitcases and bags: : Rolling suitcase',
       'Clothing, shoes and accessories: Clothes: Outerwear',
       'Electronics and appliances: Appliances: Refrigerator',
       'Furniture and interior: Furniture: Dining table',
       'Furniture and interior: Furniture: Coffee table',
       'Glasses and sunglasses: : Prescription sunglasses',
       'Home, garden and renovation: Garden and outdoor area: Greenhouse',
       'Home, garden and renovation: Garden and outdoor area: Other',
       'Watches: : Analog watch',
       'Electronics and appliances: Games and console: Game console',
       'Art and antiques: : ',
       'Furniture and interior: Furniture: Other',
       'Electronics and appliances: Appliances: Tumble dryer',
       "Baby and children's equipment: : Pram",
       'Home, garden and renovation: Bathroom equipment: Shower cabinet/wall',
       'Electronics and appliances: Various small electronics: Other',
       'Electronics and appliances: Household appliances: Other',
       'Vehicles and accessories: Other: ',
       'Furniture and interior: Furniture: Bed',
       'Electronics and appliances: Household appliances: Vacuum cleaner',
       'Sports and outdoor life: Bicycle and accessories: Bicycle',
       'Electronics and appliances: Mobile phone, tablet and accessories: Tablet',
       'Electronics and appliances: Computers and accessories: Desktop PC',
       'Electronics and appliances: Mobile phone, tablet and accessories: Other',
       'Leisure, hobby and entertainment: : Other',
       'Glasses and sunglasses: : Sunglasses without prescription',
       'Clothing, shoes and accessories: Clothes: Jacket',
       'Electronics and appliances: Appliances: Dishwasher',
       'Bags, suitcases and bags: : Suitcase',
       'Glasses and sunglasses: : Other',
       'Glasses and sunglasses: : Everyday glasses with prescription',
       'Glasses and sunglasses: : Reading glasses',
       'Electronics and appliances: Other: ',
       'Furniture and interior: Other: ',
       'Electronics and appliances: TV and sound: Head-/earphones',
       'Electronics and appliances: Appliances: Washing machine',
       'Home, garden and renovation: Other: ',
       'Electronics and appliances: Computers and accessories: Computer monitor',
       'Furniture and interior: Furniture: Sofa',
       'Electronics and appliances: Appliances: Stove/oven',
       'Watches: : Smartwatch',
       'Electronics and appliances: Mobile phone, tablet and accessories: Mobile phone',
       'Glasses and sunglasses: : Progressive',
       'Electronics and appliances: Appliances: Cooktop',
       'Electronics and appliances: Computers and accessories: Laptop',
       'Electronics and appliances: TV and sound: TV'
]

norsk_categories = ['Vesker, kofferter og bager : : Trillekoffert',
 'Klær, sko og tilbehør: Klær: Yttertøy',
 'Elektronikk og hvitevarer: Hvitevarer: Kjøleskap',
 'Møbler og interiør: Møbler: Spisebord',
 'Møbler og interiør: Møbler: Salongbord',
 'Briller og solbriller: : Solbriller med styrke',
 'Hjem, hage og oppussing: Hage og uteområde: Drivhus',
 'Hjem, hage og oppussing: Hage og uteområde: Annet',
 'Klokker: : Analog klokke',
 'Elektronikk og hvitevarer: Spill og konsoll: Spillkonsoll',
 'Kunst og antikviteter: : ',
 'Møbler og interiør: Møbler: Annet',
 'Elektronikk og hvitevarer: Hvitevarer: Tørketrommel',
 'Baby- og barneutstyr: : Barnevogn',
 'Hjem, hage og oppussing: Baderomsutstyr: Dusjkabinett/vegg',
 'Elektronikk og hvitevarer: Diverse småelektronikk: Annet',
 'Elektronikk og hvitevarer: Husholdningsapparater: Annet',
 'Kjøretøy og tilbehør: Annet: ',
 'Møbler og interiør: Møbler: Seng',
 'Elektronikk og hvitevarer: Husholdningsapparater: Støvsuger',
 'Sport og friluftsliv: Sykkel og tilbehør: Sykkel',
 'Elektronikk og hvitevarer: Mobiltelefon, nettbrett og tilbehør: Nettbrett',
 'Elektronikk og hvitevarer: Data og tilbehør: Stasjonær PC',
 'Elektronikk og hvitevarer: Mobiltelefon, nettbrett og tilbehør: Annet',
 'Fritid, hobby og underholdning: : Annet',
 'Briller og solbriller: : Solbriller uten styrke',
 'Klær, sko og tilbehør: Klær: Jakke',
 'Elektronikk og hvitevarer: Hvitevarer: Oppvaskmaskin',
 'Vesker, kofferter og bager : : Koffert',
 'Briller og solbriller: : Annet',
 'Briller og solbriller: : Hverdagsbriller med styrke',
 'Briller og solbriller: : Lesebriller',
 'Elektronikk og hvitevarer: Annet: ',
 'Møbler og interiør: Annet: ',
 'Elektronikk og hvitevarer: TV og lyd: Hode-/øretelefoner',
 'Elektronikk og hvitevarer: Hvitevarer: Vaskemaskin',
 'Hjem, hage og oppussing: Annet: ',
 'Elektronikk og hvitevarer: Data og tilbehør: Dataskjerm',
 'Møbler og interiør: Møbler: Sofa',
 'Elektronikk og hvitevarer: Hvitevarer: Komfyr/stekeovn',
 'Klokker: : Smartklokke',
 'Elektronikk og hvitevarer: Mobiltelefon, nettbrett og tilbehør: Mobiltelefon',
 'Briller og solbriller: : Progressive',
 'Elektronikk og hvitevarer: Hvitevarer: Platetopp',
 'Elektronikk og hvitevarer: Data og tilbehør: Bærbar PC',
 'Elektronikk og hvitevarer: TV og lyd: TV']

 
#Select Model
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#Select Processor
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def CLIP_MODEL(image):

    inputs = processor_clip(text=categories, images=image, return_tensors="pt", padding=True) #text = to the labels in English

    outputs = model_clip(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # Return logits as probabilities (softmax)  

    highest_prob = probs.max()
    highest_prob_index = probs.argmax()

    #find the label with the highest prob
    label = norsk_categories[highest_prob_index]
    return label

processor_GIT = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model_GIT = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

#Here we make a function that takes the image and returns the caption of the image
def GIT_MODEL(image):
    
    pixel_values = processor_GIT(images=image, return_tensors="pt").pixel_values
       
    generated_ids = model.generate(pixel_values=pixel_values, max_length=20)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_caption



st.title("Hva er bildet? :camera:")

img = st.file_uploader("Last opp et bilde", type=["png", "jpg", "jpeg"])

if img is not None:
    image = Image.open(img)

    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predikere"):
        label = CLIP_MODEL(image)
        st.write(label, )

    if st.button("Caption"):
        caption = GIT_MODEL(image)
        
        st.write(caption, )




