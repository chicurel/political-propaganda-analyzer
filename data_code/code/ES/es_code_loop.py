# --- Load Libraries ---
import numpy as np
import pandas as pd
import os
from mistralai import Mistral
import re
import spacy
from IPython.display import Markdown, display

# --- Setup ---
## Working directory
os.chdir("c:\\Users\\IPP\\Dropbox\\Test Quebec")

## API key Mistral
api_key = "tpEY0H4n2iDaRI03vPRcn4NcgqC1qCTE"
client = Mistral(api_key=api_key)

## Folder of data
folder = "1. manifesto_original\\ES"

## Get only files that end with "_es.pdf"
files = sorted(
    f for f in os.listdir(folder)
    if f.endswith("_es.pdf")
)

# LOOP throug files ----
for filename in files:
    filepath = os.path.join(folder, filename)

    # --- Upload PDF ---
    with open(filepath, "rb") as f:
        uploaded_pdf = client.files.upload(
            file={
                "file_name": filename,
                "content": f,
            },
            purpose="ocr"
        )

    # Get uploaded file id
    file_id = uploaded_pdf.id

    # --- Get PDF url ---
    signed_url = client.files.get_signed_url(file_id=file_id)
    file_url = signed_url.url
    
    
    # --- Extract text from PDF url ---
    ocr_result = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": file_url
        },
        include_image_base64=False 
    )
    
    
    # ---- Transform to DF ----
    ## Have to add region, year, party and language
    ### Join all texts
    all_text = "\n\n".join([page.markdown for page in ocr_result.pages])

    ## Save markdown
    ### Specify output folder and filename
    out_folder = "2. manifesto_ocr_md"
    md_filename = filename.replace(".pdf", ".md")
    md_path = os.path.join(out_folder, md_filename)

    ## Display markdown (Optional)
    #display(Markdown(all_text))

    ### Write to file
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(all_text)

    ## Transform to DF
    df_doc = pd.DataFrame([{
        "filename": filename,
        "text": all_text
    }])
    
    
    # ---- Clean text ----
    ## Remove titles
    df_doc["text_clean"] = df_doc["text"].str.replace(r'(?m)^#+\s+.*$', '', regex=True)

    ## Remove single words and lines with 3 or less words
    df_doc["text_clean"] = df_doc["text_clean"].str.replace(r'(?m)^(?:\S+\s*){1,3}$', '', regex=True)

    ## Remove images
    df_doc["text_clean"] = df_doc["text_clean"].str.replace(r"!\[.*?\]\(.*?\)", "", regex=True)

    ## Remove $...$ math wrappers
    df_doc["text_clean"] = df_doc["text_clean"].str.replace(r"\$", "", regex=True)

    ## Replace LaTeX \% with %
    df_doc["text_clean"] = df_doc["text_clean"].str.replace(r"\\%", "%", regex=True)

    ## Remove stray backslashes
    df_doc["text_clean"] = df_doc["text_clean"].str.replace(r"\\", "", regex=True)

    ## Remove numbers
    df_doc["text_clean"] = df_doc["text_clean"].str.replace(r"\d+", "", regex=True)

    ## Other symbols 
    ## Basically removing these: @, #, $, &, *, +, =, <, >, [, ], {, }, |, \, ~, ^, _, emojis, special symbols like ©, ®, €, £ and Uncommon accented characters
    df_doc["text_clean"] = df_doc["text_clean"].str.replace(
        r"[^a-zA-Z0-9À-ÖØ-öø-ÿ.,;:!?'()\-\s]", 
        "",
        regex=True
    )

    ## Collapse multiple spaces into one
    df_doc["text_clean"] = df_doc["text_clean"].str.replace(r"\s{2,}", " ", regex=True)

    ## Trim leading/trailing spaces
    df_doc["text_clean"] = df_doc["text_clean"].str.strip()

    ## Remove line breaks
    df_doc["text_clean"] = df_doc["text_clean"].str.replace(r"(?m)\n", " ", regex=True)

    ## Remove repeated punctuation (.,)
    df_doc["text_clean"] = df_doc["text_clean"].str.replace(r"(\.\%|\(\)|[.,\-()])(?:\s*\1){1,}", r"\1", regex=True)
    
    
    # ---- Save cleaned text to file ----
    out_folder = "3. manifesto_ocr_txt_clean"
    txt_filename = df_doc["filename"].iloc[0].replace(".pdf", "_txt_clean.txt")
    txt_path = os.path.join(out_folder, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(df_doc["text_clean"].iloc[0])
        
        
    # ---- Add Region, Year and Party ----
    pattern = r"^([A-Z]+)_([0-9]{4})_([A-Z]+)_([a-z]+)\.pdf$"

    match = re.match(pattern, filename)

    if match:
        region, year, party, language = match.groups()
    else:
        raise ValueError("Filename does not match expected pattern")


    df_doc["region"] = region
    df_doc["year"] = year
    df_doc["party"] = party
    df_doc["language"] = language
    
    
    # ---- Divide DF by lines using spaCy----
    ## Load Spanish spaCy model once
    nlp = spacy.load("es_core_news_md")

    ## Divide lines
    def split_sentences(text):
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip() != ""]

    df_lines = df_doc.assign(
        line=df_doc["text_clean"].apply(split_sentences)
    ).explode("line")

    df_lines = df_lines.reset_index(drop=True)

    ## Remove blank sentences
    df_lines = df_lines[df_lines["line"].str.strip() != ""]

    ## Remove sentences with 3 or fewer words
    df_lines = df_lines[df_lines["line"].apply(lambda s: len(s.split()) > 3)]
    df_lines = df_lines.reset_index(drop=True)

    ## Trim leading/trailing spaces
    df_lines["line"] = df_lines["line"].str.strip()

    ## Add number of words column
    df_lines["num_words"] = df_lines["line"].str.split().str.len()
    
    
    # ---- Save dataframe no translate ----
    ## Select columns
    df_lines = df_lines[["filename", "region","year","party","language","line","num_words"]]

    base_name = filename.replace(".pdf", "")
    df_lines.to_csv(f"4. manifesto_df/{base_name}_df.csv", index=False)
    df_lines.to_csv(f"5. manifesto_translated/{base_name}_df_translated.csv", index=False)
    
    print(f"Processed {filename}: {file_url}")
