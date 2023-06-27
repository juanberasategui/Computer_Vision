
import streamlit as st
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
#from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

categories = ['Electronics and appliances', 'Watches', 'Home, garden and renovation', 'Glasses and sunglasses',
        'Furniture and interior', 'Sports and outdoor life', 'Clothing, shoes and accessories', 'Bags, suitcases and bags',
        'Jewelry and similar items', 'Leisure, hobby and entertainment', 'Baby and childrens equipment',
        'Vehicles and accessories', 'Personal care', 'Pet supplies', 'Art and antiques', 'Food and drinks', 'Cash, wallet and gift cards']

norsk_categories = ['Elektronikk og hvitevarer', 'Klokker', 'Hjem, hage og oppussing',
       'Briller og solbriller', 'Møbler og interiør',
       'Sport og friluftsliv', 'Klær, sko og tilbehør',
       'Vesker, kofferter og bager ', 'Smykker og lignende',
       'Fritid, hobby og underholdning', 'Baby- og barneutstyr',
       'Kjøretøy og tilbehør', 'Personlig pleie', 'Dyreutstyr',
       'Kunst og antikviteter', 'Mat og drikke',
       'Kontanter, lommebok og gavekort']

cat_dict = {'Elektronikk og hvitevarer': ['Hvitevarer', 'Diverse småelektronikk', 'Spill og konsoll',
                                           'Mobiltelefon, nettbrett og tilbehør', 'TV og lyd', 
                                             'Sport og friluft elektronikk', 'Foto og video', 'Husholdningsapparater',
                                               'Data og tilbehør'], 'Klokker': [None], 'Hjem, hage og oppussing': 
                                               ['Kjøkkenutstyr ','Sikkerhet og alarm', 'Soveromsutstyr', 'Byggevarer',
                                                 'Oppvarming og ventilasjon', 'Verktøy og tilbehør', 'Baderomsutstyr', 
                                                   'Hage og uteområde'], 'Briller og solbriller': [None], 'Møbler og interiør':
                                                     ['Tepper og tekstiler', 'Møbler', 'Dekorasjon og pyntegjenstander',
                                                       'Lamper/belysning'], 'Sport og friluftsliv': ['Vannsport', 'Trimapparat/styrkeutstyr',
                                                        'Våpen og tilbehør', 'Vintersport', 'Sykkel og tilbehør',  'Sport og ballspill', 'Jakt, fiske og friluft'], 'Klær, sko og tilbehør': ['Klær', 'Sko', 'Tilbehør', 'Hodeplagg'], 'Vesker, kofferter og bager ': [None], 'Smykker og lignende': [None], 'Fritid, hobby og underholdning': [None], 'Baby- og barneutstyr': [None], 'Kjøretøy og tilbehør': ['Tilbehør til bil', 'Tilbehør til båt', 'Tilbehør til caravan', 'Tilbehør til MC'], 'Personlig pleie': [None], 'Dyreutstyr': [None], 'Kunst og antikviteter': [None], 'Mat og drikke': [None], 'Kontanter, lommebok og gavekort': ['Kontanter', 'Lommebok']}
 

cat_dict_english = {
    'Electronics and appliances': ['Appliances', 'Miscellaneous Electronics', 'Games and Consoles',
                                   'Mobile Phones, Tablets, and Accessories', 'TV and Audio', 'Other',
                                   'Sports and Outdoor Electronics', 'Photo and Video', 'Household Appliances',
                                   'Computers and Accessories'],
    'Watches': [None],
    'Home, garden and renovation': ['Kitchen Equipment', 'Security and Alarm', 'Bedroom Equipment', 'Building Materials',
                                    'Heating and Ventilation', 'Tools and Accessories', 'Bathroom Equipment', 'Other',
                                    'Garden and Outdoor Area'],
    'Glasses and sunglasses': [None],
    'Furniture and interior': ['Carpets and Textiles', 'Furniture', 'Decoration and Ornaments', 'Other', 'Lamps/Lighting'],
    'Sports and outdoor life': ['Water Sports', 'Exercise Equipment', 'Weapons and Accessories', 'Winter Sports',
                                'Bicycles and Accessories', 'Other', 'Sports and Ball Games', 'Hunting, Fishing, and Outdoor'],
    'Clothing, shoes and accessories': ['Clothing', 'Other', 'Shoes', 'Accessories', 'Headwear'],
    'Bags, suitcases and bags': [None],
    'Jewelry and similar items': [None],
    'Leisure, hobby and entertainment': [None],
    'Baby and childrens equipment': [None],
    'Vehicles and accessories': ['Car Accessories', 'Boat Accessories', 'Other', 'Caravan Accessories', 'Motorcycle Accessories'],
    'Personal care': [None],
    'Pet supplies': [None],
    'Art and antiques': [None],
    'Food and drinks': [None],
    'Cash, wallet and gift cards': ['Cash', 'Wallet']
}

#Select Model
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#Select Processor
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def CLIP_MODEL(image, categories, norsk_categories):
    inputs = processor_clip(text=categories, images=image, return_tensors="pt", padding=True)
    outputs = model_clip(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    top_indices = sorted_indices[0][:3]  # Get the indices of the top three probabilities
    
    top_categories = [norsk_categories[index] for index in top_indices]  # Get the corresponding categories
    
    return top_categories, top_indices

#processor_GIT = AutoProcessor.from_pretrained("microsoft/git-large-r-coco")
#model_GIT = AutoModelForCausalLM.from_pretrained("microsoft/git-large-r-coco")

#Here we make a function that takes the image and returns the caption of the image
#def GIT_MODEL(image):
    
    pixel_values = processor_GIT(images=image, return_tensors="pt").pixel_values
       
    generated_ids = model_GIT.generate(pixel_values=pixel_values, max_length=20)
    generated_caption = processor_GIT.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_caption




#ARCHITECTURE OF THE APP

buttons = ["button1", "button2", "button3", "button4", "button5", "button6", "button7", "button8", "button9"]

if "predikere" not in st.session_state:
    st.session_state["predikere"] = False

for button in buttons:
    if button not in st.session_state:
        st.session_state[button] = False


st.title("Hva er bildet? :camera: :scissors: ")

img = st.file_uploader("Last opp et bilde", type=["png", "jpg", "jpeg"])

if img is not None:
    image = Image.open(img)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predikere"):
        st.session_state["predikere"] = not st.session_state["predikere"]

    if st.session_state["predikere"]:
        labels, indices = CLIP_MODEL(image, categories, norsk_categories)
        if st.button(str(labels[0])):
            st.session_state["button1"] = not st.session_state["button1"]
        if st.button(str(labels[1])):
            st.session_state["button2"] = not st.session_state["button2"]
        if st.button(str(labels[2])):
            st.session_state["button3"] = not st.session_state["button3"]
    
    if st.session_state["predikere"] and st.session_state["button1"]:
        st.write("Du har valgt: " + str(labels[0]) + ", velg en underkategori")
        eng_category = categories[int(indices[0])]
        norsk_category = norsk_categories[int(indices[0])]
        categories = cat_dict_english[eng_category]
        norsk_categories = cat_dict[norsk_category]
        labels2, indices2 = CLIP_MODEL(image, categories, norsk_categories)
        if st.button(str(labels2[0])):
            st.session_state["button4"] = not st.session_state["button4"]
        if st.button(str(labels2[1])):
            st.session_state["button5"] = not st.session_state["button5"]
    
    if st.session_state["predikere"] and st.session_state["button2"]:
        st.write("Du har valgt: " + str(labels[1]) + ", velg en underkategori")
        eng_category = categories[int(indices[1])]
        norsk_category = norsk_categories[int(indices[1])]
        categories = cat_dict_english[eng_category]
        norsk_categories = cat_dict[norsk_category]
        labels2, indices = CLIP_MODEL(image, categories, norsk_categories)
        if st.button(str(labels2[0])):
            st.session_state["button6"] = not st.session_state["button6"]
        if st.button(str(labels[1])):
            st.session_state["button7"] = not st.session_state["button7"]
    
    if st.session_state["predikere"] and st.session_state["button3"]:
        st.write("Du har valgt: " + str(labels[2]) + ", velg en underkategori")
        eng_category = categories[int(indices[2])]
        norsk_category = norsk_categories[int(indices[2])]
        categories = cat_dict_english[eng_category]
        norsk_categories = cat_dict[norsk_category]
        labels2, indices = CLIP_MODEL(image, categories, norsk_categories)
        if st.button(str(labels2[0])):
            st.session_state["button8"] = not st.session_state["button8"]
        if st.button(str(labels2[1])):
            st.session_state["button9"] = not st.session_state["button9"]

    
    if st.session_state["predikere"] and st.session_state["button1"] and st.session_state["button4"]:
        st.write("Du har valgt: "+str(labels[0])+ "og"+ str(labels2[0]) + ", velg en underkategori")
        st.write("Takk vi er ferdig")
    
    if st.session_state["predikere"] and st.session_state["button1"] and st.session_state["button5"]:
        st.write("Du har valgt: "+str(labels[0])+ "og"+ str(labels2[1]) + ", velg en underkategori")
        st.write("Takk vi er ferdig")
    
    if st.session_state["predikere"] and st.session_state["button2"] and st.session_state["button6"]:
        st.write("Du har valgt: "+str(labels[1])+ "og"+ str(labels2[0]) + ", velg en underkategori")
        st.write("Takk vi er ferdig")

    if st.session_state["predikere"] and st.session_state["button2"] and st.session_state["button7"]:
        st.write("Du har valgt: "+str(labels[1])+ "og"+ str(labels2[1]) + ", velg en underkategori")
        st.write("Takk vi er ferdig")

    if st.session_state["predikere"] and st.session_state["button3"] and st.session_state["button8"]:
        st.write("Du har valgt: "+str(labels[2])+ "og"+ str(labels2[0]) + ", velg en underkategori")
        st.write("Takk vi er ferdig")
    
    if st.session_state["predikere"] and st.session_state["button3"] and st.session_state["button9"]:
        st.write("Du har valgt: "+str(labels[2])+ "og"+ str(labels2[1]) + ", velg en underkategori")
        st.write("Takk vi er ferdig")
