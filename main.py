# ----------------------------------------------------------------------------------------------------------------
# Task 2: Identifying alternative materials
# Author: Simon Markthaler
# Date: 2025-07-19
# Version: 0.0.1
#
# main:
# > Main script unsupervised similarity matching of materials.
# ----------------------------------------------------------------------------------------------------------------

# --------------------------------------------Import Python libraries---------------------------------------------
import os

from src.model import SimilarityModel
from src.preprocessing import Preprocessing
from src.utils import Configuration

#TODO: Pyyaml in requirements auch in anderen
#TODO: Describe how I used a GPT (Github Copilot) to analyse the column PART_DESCRIPTION for any materials not present in the material list
#TODO: Delete second algorithm in BinClass
#TODO: Include GPT for checking for unknown Materials in part descriptions

if __name__ == "__main__":
    Config = Configuration()                                    # Load configuration
    Config.path = os.path.dirname(__file__)

    Preproc = Preprocessing(Config)             # Preprocessing class for loading and preprocessing data sets   

    df_data = Preproc.load_data()   # Load data set

    df_data, fuse_material_list = Preproc.matching(df_data)

    model = SimilarityModel()                                   # Model for similarity analysis

    # Create TF-IDF Matrix for part descriptions which ascribes a weight to each term based on its frequency in the document and across all documents
    tfidf_matrix = model.tf_idf(df_data, fuse_material_list)    

    # Create material vectors by averaging the TF-IDF vectors of all part descriptions that belong to each material
    mat_vectors, materials = model.create_material_vectors(tfidf_matrix)

    # Compute cosine similarity
    model.cosine_sim(mat_vectors, materials)
