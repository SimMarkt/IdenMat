"""
---------------------------------------------------------------------------------------------------
IdenMat: Identifying alternative battery electrode materials
         via unsupervised similarity matching (NLP)
GitHub Repository: https://github.com/SimMarkt/IdenMat.git

model:
> Provides the similarity model containing TF-IDF and cosine similarity.
---------------------------------------------------------------------------------------------------
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack, csr_matrix

from src.utils import Configuration, plot_cosine_similarity_heatmap

class SimilarityModel:
    """SimilarityModel class for unsupervised similarity matching of materials."""
    def __init__(self, config: Configuration) -> None:
        """
        :param config: Configuration object containing environment variables.
        """
        self.config = config

        self.path_results = self.config.path + self.config.data_path_results

        self.df_data = None
        self.bat_material_list = None
        self.materials = None

        print("\nStarting material similarity matching...")

    def tf_idf(self, df_data: pd.DataFrame, bat_material_list):
        """        
        Create TF-IDF matrix for part descriptions.
        :param df_data: DataFrame containing the cleaned data.
        :param bat_material_list: List of all electrode materials in
                                  the 'ELECTRODE_MATERIAL' column.
        :return tfidf_matrix: TF-IDF matrix.
        """

        self.df_data = df_data
        self.bat_material_list = bat_material_list
        # Drop all samples without Part description which includes a ELECTRODE_MATERIAL
        # mask: True if PART_DESCRIPTION contains any ELECTRODE_MATERIAL
        mask = self.df_data.apply(self.contains_material, axis=1)
        self.df_data = self.df_data[mask].reset_index(drop=True)          # Drop the samples

        print(f'...Data shape after droping undefined samples: {self.df_data.shape}')

        # Vectorize the part descriptions
        print("...Vectorize the part descriptions using TF-IDF")
        # Computes:
        # [Frequency of a term (TF) in a description]
        #  x
        # [How rare the term is across all descriptions]
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.df_data['PART_DESCRIPTION'].fillna(""))

        # print(tfidf.get_feature_names_out())

        # Print tdidf structure as a dataframe to the TUI
        # df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
        # print(df_tfidf)

        return tfidf_matrix

    def contains_material(self, row) -> bool:
        """        
        Check if PART_DESCRIPTION contains any ELECTRODE_MATERIAL.
        :param row: DataFrame row.
        :return: True if PART_DESCRIPTION contains any ELECTRODE_MATERIAL, else False.
        """
        description = row['PART_DESCRIPTION']
        if pd.isna(description) or str(description).strip() == '':
            return False
        if not any(mat in description for mat in self.bat_material_list):
            return False
        return True

    def create_material_vectors(self, tfidf_matrix):
        """
        Create material vectors by averaging the TF-IDF vectors of all part descriptions
        that belong to each material.
        :param tfidf_matrix: TF-IDF matrix.
        :return material_matrix: Matrix of material vectors.
        """
        print("...Group dataframe by materials")
        # Group rows by material
        df_grouped = self.df_data.groupby('ELECTRODE_MATERIAL')

        # Obtain a single vector that represents each material
        # (e.g., "Nickel Manganese Cobalt", "Lithium Iron Phosphate")
        # by averaging the TF-IDF vectors of all part descriptions that belong to that material
        material_vectors = {}   # dictionary for storing the vectors to every material
        for material, group in df_grouped:
            idxs = group.index
            avg_vector = tfidf_matrix[idxs].mean(axis=0)  # average vector across rows
            avg_vector = csr_matrix(avg_vector)  # Convert to 2D sparse matrix
            material_vectors[material] = avg_vector

        # Convert to a matrix of material vectors
        materials = list(material_vectors.keys())
        material_matrix = vstack([material_vectors[m] for m in materials])

        print("...Completed vectorization.")

        return material_matrix, materials

    def cosine_sim(self, material_matrix, materials):
        """        
        Compute cosine similarity between material vectors.
        """

        print("...Calculate cosine similarity")

        # Compute the cosine of the angle between each pair of mat_vectors
        cosine_sim = cosine_similarity(material_matrix)
        # Represents a similarity matrix [i][j] with similarity score
        # between material i and material j

        top_k = {}
        for i, mat in enumerate(materials):
            sims = list(enumerate(cosine_sim[i]))
            sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
            top_5 = [materials[j] for j, score in sims_sorted[1:6]]
            top_k[mat] = top_5

        print("...Plot similarity heat map")
        sim_df = pd.DataFrame(cosine_sim, index=materials, columns=materials)
        plot_cosine_similarity_heatmap(sim_df,self.path_results)

        print("...Results: Top 5 most similar materials")
        for material, similar_materials in top_k.items():
            print(f"     {material} -> {', '.join(similar_materials)}")

        print("\nFinished similarity matching.")
        print("-----------------------------------------------------------------------------")
