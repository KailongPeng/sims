def visualize_prob():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the TSV file
    file_path = '/gpfs/milgram/scratch60/turk-browne/kp578/chanales/v1rf/probes.tsv'
    probes_df = pd.read_csv(file_path, sep='\t')

    # Function to extract and reshape matrix data
    def extract_and_reshape(matrix_data, prefix):
        reshaped_matrix = np.zeros((12, 12))
        for i in range(12):
            for j in range(12):
                column_name = f'{prefix}[2:{i},{j}]'
                if column_name in matrix_data:
                    reshaped_matrix[i, j] = matrix_data[column_name]
        return reshaped_matrix

    # Prepare a dictionary to hold matrices for each category and type (%LGNon and %LGNoff)
    matrices = {}

    # Define categories and types
    categories = ['H', 'L', 'V', 'R']
    types = ['%LGNon', '%LGNoff']

    # Iterate over categories and types to extract and store matrices
    for category in categories:
        matrices[category] = {}
        for type_ in types:
            category_data = probes_df[probes_df['$Name'] == category]
            if not category_data.empty:
                category_dict = category_data.iloc[0].to_dict()
                matrices[category][type_] = extract_and_reshape(category_dict, type_)

    # Visualize each matrix pair (%LGNon and %LGNoff) side by side for each category
    fig, axes = plt.subplots(len(categories), 2, figsize=(10, 20))  # Adjust figsize as needed

    for i, category in enumerate(categories):
        for j, type_ in enumerate(types):
            ax = axes[i, j]
            cax = ax.matshow(matrices[category][type_], cmap='viridis')
            ax.set_title(f'{category} - {type_}')
            ax.set_xticks(range(12))
            ax.set_yticks(range(12))
            ax.set_xticklabels(range(1, 13))
            ax.set_yticklabels(range(1, 13))
            fig.colorbar(cax, ax=ax, orientation='vertical')

    plt.tight_layout()
    plt.show()


import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("/gpfs/milgram/scratch60/turk-browne/kp578/chanales/v1rf/record0304_1_.csv", sep='\t')

# Find columns that begin with the specified prefixes
v1actm_cols = [col for col in df.columns if col.startswith("#V1ActM")]
lgnonact_cols = [col for col in df.columns if col.startswith("#LGNonAct")]
lgnoffact_cols = [col for col in df.columns if col.startswith("#LGNoffAct")]

# Merge the columns by summing their values
df['V1ActM'] = df[v1actm_cols].values.tolist()
df['LGNonAct'] = df[lgnonact_cols].values.tolist()
df['LGNoffAct'] = df[lgnoffact_cols].values.tolist()

# Drop the original columns
df.drop(v1actm_cols + lgnonact_cols + lgnoffact_cols, axis=1, inplace=True)

# extract the V1ActM, for the $TrialName L
def extract_columns(df, trial_name, column_name):
    ll = df[df['$TrialName'] == trial_name][column_name]
    # reindex the list of lists
    ll.index = range(len(ll))
    ll_array = np.zeros((len(ll), len(ll[0])))
    for l in ll:
        ll_array += l

    return ll_array

# Extract the V1ActM for the specified trial
v1actm_H = extract_columns(df, "H", 'V1ActM')
v1actm_L = extract_columns(df, "L", 'V1ActM')
v1actm_V = extract_columns(df, "V", 'V1ActM')
v1actm_R = extract_columns(df, "R", 'V1ActM')

print(f"v1actm_L.shape: {v1actm_L.shape}")

# I have four matrices whose shape is (5, 196). I want to

