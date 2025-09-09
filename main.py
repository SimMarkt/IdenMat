"""
---------------------------------------------------------------------------------------------------
IdenMat: Identifying alternative battery electrode materials 
         via unsupervised similarity matching (NLP)
GitHub Repository: https://github.com/SimMarkt/IdenMat.git

main:
> Main script for unsupervised similarity matching of materials.
---------------------------------------------------------------------------------------------------
"""

# --------------------------------------------Import Python libraries---------------------------------------------
import os

from src.model import SimilarityModel
from src.preprocessing import Preprocessing
from src.utils import Configuration

if __name__ == "__main__":
    Config = Configuration()                                            # Load configuration
    Config.path = os.path.dirname(__file__)

    Preproc = Preprocessing(Config)                                     # Preprocessing class for loading and preprocessing data sets   

    df_data = Preproc.load_data()                                       # Load data set

    df_data, fuse_material_list = Preproc.matching(df_data)             # Match 'ELECTRODE_MATERIAL' with 'PART_DESCRIPTION' and infer missing values.

    model = SimilarityModel(Config)                                     # Model for similarity analysis

    # Create TF-IDF Matrix for part descriptions which ascribes a weight to each term based on its frequency in the document and across all documents
    tfidf_matrix = model.tf_idf(df_data, fuse_material_list)    

    # Create material vectors by averaging the TF-IDF vectors of all part descriptions that belong to each material
    material_matrix, materials = model.create_material_vectors(tfidf_matrix)

    # Compute cosine similarity
    model.cosine_sim(material_matrix, materials)
