import easyocr
import cv2
import requests
import re
from PIL import Image
import streamlit as st
import numpy as np



api_key = st.secrets["api_key"]

# Load the EasyOCR reader
reader = easyocr.Reader(['en'])

API_URL = "https://api-inference.huggingface.co/models/flair/ner-english-large"
headers = {"Authorization": "Bearer {api_key}"}

## Image uploading function ##
def image_upload_and_ocr(reader, uploaded_file):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.resize((640, 480))
        
        image_np = np.array(image)  # Convert image to NumPy array
        result2 = reader.readtext(image_np)
        texts = [item[1] for item in result2]
        result = ' '.join(texts)
        
        return result2, result, image
    else:
        return None, None, None

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def get_ner_from_transformer(output):
    data = output
    named_entities = {}
    for entity in data:
        entity_type = entity['entity_group']
        entity_text = entity['word']
        
        if entity_type not in named_entities:
            named_entities[entity_type] = []
        
        named_entities[entity_type].append(entity_text)
    
    return entity_type, named_entities

def drawing_detection(res2, image):
    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Draw bounding boxes around the detected text regions
    for detection in res2:
        # Extract the bounding box coordinates
        points = detection[0]  # List of points defining the bounding box
        x1, y1 = int(points[0][0]), int(points[0][1])  # Top-left corner
        x2, y2 = int(points[2][0]), int(points[2][1])  # Bottom-right corner
        
        # Draw the bounding box
        cv2.rectangle(cv2_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
        # Add the detected text
        text = detection[1]
        cv2.putText(cv2_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    st.image(cv2_image, caption='Detected text on the card', width=710)
    return cv2_image

# Function to extract phone numbers from text using regular expression
def extract_phone_numbers(text):
    # Regular expression pattern for detecting phone numbers
    PHONE_PATTERN = r'(?:ph|phone|phno)?\s*(?:[+-]?\d\s*[\(\)]*){7,}'

    # Find phone numbers using regular expression
    phone_numbers = re.findall(PHONE_PATTERN, text, re.IGNORECASE)
    # Return the extracted phone numbers
    return phone_numbers or None

# Function to extract email addresses from text using regular expression
def extract_email(text):
    emails = []
    # Regular expression pattern for detecting email addresses with variations
    reg = r'[a-z0-9_.-]+(?:\s*@\s*)[a-z]+(?:\s*\.?\s*[a-z]{2,3})\s*'
    # Find email addresses using regular expression
    res = re.findall(reg, text, re.IGNORECASE)
    # Print the extracted email addresses
    for email in res:
        emails.append(email.strip())
    return emails or None

# Function to extract designations from text using regular expression
def extract_designation(text):
    designations = []
    # Regular expression pattern for detecting designations
    designation_regex = r'\b(?:CEO|CFO|CTO|COO|CMO|CIO|President|Vice\s?President|Director|Manager|Executive\s?Director|Assistant\s?Manager|Account\s?Manager|Sales\s?Manager|Marketing\s?Manager|Product\s?Manager|Project\s?Manager|HR\s?Manager|Human\s?Resources\s?Manager|Operations\s?Manager|Business\s?Development\s?Manager|Senior\s?Manager|General\s?Manager|Team\s?Lead|Consultant|Analyst|Engineer|Architect|Designer|Developer|Programmer|Coordinator|Specialist|Supervisor|Administrator|Assistant|Associate|Partner|Founder|Owner|Principal|Expert|Technician|Officer|Representative|Agent|Accountant|Auditor|Trainer|Coach|Educator|Professor|Instructor|Researcher|Scientist|Doctor|Nurse|Therapist|Pharmacist|Attorney|Lawyer|Legal\s?Counsel|Paralegal|Advocate|Solicitor|Notary|Financial\s?Advisor|Investment\s?Advisor|Wealth\s?Manager|Broker|Realtor|Mortgage\s?Broker|Insurance\s?Agent)\b'

    # Find designations using regular expression
    designations = re.findall(designation_regex, text, re.IGNORECASE)

    return designations or None

# Function to extract website URLs from text using regular expression
def extract_websites(text):
    websites_found=[]
    pattern = r'(https?://)?(www\.)?(\w+)(\.\w+)+'
    websites = re.findall(pattern, text)
    return ["".join(website) for website in websites] or None

# Function to extract PIN codes from text using regular expression
def extract_pin_code(text):
    pin_code_pattern = r'\b\d{6}\b'
    pin_code_match = re.search(pin_code_pattern, text.lower())
    
    # Retrieve the PIN code if found
    if pin_code_match:
        pin_code = pin_code_match.group()
        return pin_code
    else:
        return None

import pandas as pd

# Streamlit UI
st.title("Business Card Data Extractor using OpenCV and Streamlit")

uploaded_file = st.file_uploader(label="Please upload a business card", type=['jpeg', 'jpg', 'png', 'webp'], accept_multiple_files=False)

if uploaded_file is not None:
    res2, res, image = image_upload_and_ocr(reader, uploaded_file)
    
    if res2 is not None:
        drawing_image = drawing_detection(res2, image)

        try:
            output = query({
                "inputs": res,
            })

            entity_type, named_entities = get_ner_from_transformer(output)
        except Exception as e:
            st.error("An error occurred while processing the business card. Please try again later.")
            st.error(f"Error details: {str(e)}")

        extracted_data = {}

        # Function to extract person's name
        # Assuming the person's name is extracted by NER
        names = named_entities.get("PER", [])
        if names:
            selected_name = st.selectbox("Select Person's Name:", [""] + names)
            if selected_name:
                extracted_data["Name"] = selected_name
            else:
                manual_name = st.text_input("Enter Person's Name manually:")
                if manual_name:
                    extracted_data["Name"] = manual_name

        # Function to extract designations
        designations = extract_designation(res)
        if designations is not None:
            selected_designation = st.selectbox("Select Designation:", [""] + designations)
            if selected_designation:
                extracted_data["Designation"] = selected_designation
            else:
                manual_designation = st.text_input("Enter Designation manually:")
                if manual_designation:
                    extracted_data["Designation"] = manual_designation

        # Function to extract company names
        # Assuming the organization names extracted by NER represent company names
        company_names = named_entities.get("ORG", [])
        if company_names:
            selected_company_name = st.selectbox("Select Company Name:", [""] + company_names)
            if selected_company_name:
                extracted_data["Company Name"] = selected_company_name
            else:
                manual_company_name = st.text_input("Enter Company Name manually:")
                if manual_company_name:
                    extracted_data["Company Name"] = manual_company_name

        # Function to extract email addresses
        emails = extract_email(res)
        if emails is not None:
            selected_email = st.selectbox("Select Email:", [""] + emails)
            if selected_email:
                extracted_data["Email"] = selected_email
            else:
                manual_email = st.text_input("Enter Email manually:")
                if manual_email:
                    extracted_data["Email"] = manual_email

        # Function to extract website URLs
        websites = extract_websites(res)
        if websites is not None:
            selected_website = st.selectbox("Select Website:", [""] + websites)
            if selected_website:
                extracted_data["Website"] = selected_website
            else:
                manual_website = st.text_input("Enter Website manually:")
                if manual_website:
                    extracted_data["Website"] = manual_website

        # Function to extract phone numbers
        phone_numbers = extract_phone_numbers(res)
        if phone_numbers is not None:
            selected_phone_number = st.selectbox("Select Phone Number:", [""] + phone_numbers)
            if selected_phone_number:
                extracted_data["Phone Number"] = selected_phone_number
            else:
                manual_phone_number = st.text_input("Enter Phone Number manually:")
                if manual_phone_number:
                    extracted_data["Phone Number"] = manual_phone_number

       # Concatenate all the text returned by the API for location
        locations = named_entities.get("LOC", [])
        if locations:
            concatenated_location = ", ".join(locations)
            selected_location = st.selectbox("Select Location:", [""] + [concatenated_location])
            if selected_location:
                extracted_data["Location"] = selected_location
            else:
                manual_location = st.text_input("Enter Location manually:")
                if manual_location:
                    extracted_data["Location"] = manual_location
        else:
            manual_location = st.text_input("Enter Location manually:")
            if manual_location:
                extracted_data["Location"] = manual_location


        # Function to extract PIN codes
        pin_code = extract_pin_code(res)
        if pin_code is not None:
            selected_pin_code = st.selectbox("Select PIN Code:", ["", pin_code])
            if selected_pin_code:
                extracted_data["PIN Code"] = selected_pin_code
            else:
                manual_pin_code = st.text_input("Enter PIN Code manually:")
                if manual_pin_code:
                    extracted_data["PIN Code"] = manual_pin_code

        # Display extracted data
        if extracted_data:
            st.write("Extracted Data:")
            df = pd.DataFrame([extracted_data], columns=["Name", "Designation", "Company Name", "Email", "Website", "Phone Number", "Location", "PIN Code"])
            st.write(df)
