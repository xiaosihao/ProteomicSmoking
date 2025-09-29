##Protein annotation
#-------------------
import pandas as pd  # import libraries
import requests
from tqdm import tqdm
#read the data from the csv file
proteins = pd.read_csv('file_name')['Proteins'].to_list()  # read data

# Replace 'your_app_name' with a name for your app when registering with STRING
caller_identity = 'your_app_name'
api_key = 'your_api_key'  # Optional: Include if you have an API key
species = '9606'  # Human, Homo sapiens

def get_annotations(protein):
    url = f"https://string-db.org/api/json/get_string_ids?identifiers={protein}&species={species}&caller_identity={caller_identity}&echo_query=1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Extract annotation if it exists in the response
        annotation_full = data[0].get('annotation') if data and 'annotation' in data[0] else 'No annotation found'
        # Keep only the first two sentences
        annotation = ' '.join(annotation_full.split('. ')[:2])
        return annotation
    else:
        return 'Failed to fetch data'

# Using a dictionary to store the results
results = {'Protein': [], 'Annotation': []}

for protein in tqdm(proteins):
    if protein == 'EBI3_IL27':
        protein = 'EBI3'
    annotation = get_annotations(protein)
    results['Protein'].append(protein)
    results['Annotation'].append(annotation)

# Convert the results into a DataFrame
df = pd.DataFrame(results)
#sort by protein names
df = df.sort_values('Protein')
df.to_csv('file_name',index=False)  # save to csv