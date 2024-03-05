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

    """
    Two ways to solve the current issue:
        One is to change the training and testing paradigm to be train some first test train a lot and then test
        The other way is to drastically increase the number of trials in the test probes.
            Create a python GUI so that I can draw in a 12x12 matrix whose values are 0 or 1. The save the matrix for further use.
            Create a simple GUI for drawing in a 12x12 matrix and saving the matrix values (0 or 1).
            Create a simple way for me to draw in a 12x12 matrix and saving the matrix values (0 or 1).

        
    """

# def new_probes():
#     new_prob = "/gpfs/milgram/scratch60/turk-browne/kp578/chanales/v1rf/matrix.npy"
#     new_prob = np.load(new_prob)  # 5x12x12
#     # append these new probes to the old probes, the new_prob $Name is "X1" to "X5", then save the new probes to a new probes_new.tsv
def append_new_probes():
    import pandas as pd
    import numpy as np
    # Load the old probes
    file_path = '/gpfs/milgram/scratch60/turk-browne/kp578/chanales/v1rf/probes.tsv'  # Adjust the path as necessary
    probes_df = pd.read_csv(file_path, sep='\t')

    # Load the new probe matrices
    new_prob_path = "/gpfs/milgram/scratch60/turk-browne/kp578/chanales/v1rf/matrix.npy"  # Adjust the path as necessary
    new_prob_matrices = np.load(new_prob_path)  # Assuming this is 5x12x12

    # Generate new probe rows
    for i, matrix in enumerate(new_prob_matrices):
        # Initialize the new row with the name and then placeholder values for each grid point
        new_row = [f'X{i + 1}'] + list(matrix.flatten())

        # Append the new row to the DataFrame
        probes_df = pd.concat([probes_df, pd.DataFrame([new_row])], ignore_index=True)

    # Define the new header based on the original structure, adjusting for any non-matrix columns if necessary
    header = ['$Name'] + [f'%LGNon[2:{i},{j}]' for i in range(12) for j in
                          range(12)]  # Adjust if you have both LGNon and LGNoff

    # Save the updated dataframe to a new TSV file
    new_file_path = '/gpfs/milgram/scratch60/turk-browne/kp578/chanales/v1rf/probes_new.tsv'
    probes_df.to_csv(new_file_path, sep='\t', index=False, header=header)


append_new_probes()


import pandas as pd
import numpy as np

# Load the dataset
# path_name = "/gpfs/milgram/scratch60/turk-browne/kp578/chanales/v1rf/record0304_1_.csv"
path_name = "/gpfs/milgram/scratch60/turk-browne/kp578/chanales/v1rf/record0305_epc100trl100.csv"
df = pd.read_csv(path_name, sep='\t')
# delete the first 4 rows
df = df.iloc[4:]

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
    for ii, l in enumerate(ll):
        ll_array[ii, :] = l

    return ll_array

# Extract the V1ActM for the specified trial
layerName = "LGNonAct"  # or "LGNoffAct" or "V1ActM"
H_act = extract_columns(df, "H", layerName)
L_act = extract_columns(df, "L", layerName)
V_act = extract_columns(df, "V", layerName)
R_act = extract_columns(df, "R", layerName)

print(f"H_act.shape: {H_act.shape}")

"""
I have four matrices whose shape is (2, 196). Their names are v1actm_H, v1actm_L, v1actm_V, and v1actm_R, corresponding to the stimuli H, L, V, and R, respectively.
(2, 196) means that each matrix has 2 rows and 196 columns, corresponding to 2 trials, one before learning and one after learning.
196 is the number of neurons in the V1 region.

Now I will use correlation to compare the similarity between each stimuli H, L, V, and R.
This way for before learning and after learning, I will two 4 x 4 matrices, each of which will represent the similarity between the stimuli representations.
Now use the difference between the two matrices (after minus before) as a measure of learning or y axis of the learning curve, in other words, difference matrix is 4x4, reshape it to 1x16 and plot it as the y axis of the learning curve. 
Use the before learning matrix as the x axis of the learning curve. In other words, reshape the before learning matrix to 1x16 and plot it as the x axis of the learning curve.
Now plot the learning curve.
 
code the whole process in a function called `rep_NMPH` whose input is v1actm_H, v1actm_L, v1actm_V, and v1actm_R.
"""
import numpy as np
import matplotlib.pyplot as plt


def compute_correlation_matrix(stimuli):
    """Compute the correlation matrix for a set of stimuli representations."""
    num_stimuli = len(stimuli)
    correlation_matrix = np.zeros((num_stimuli, num_stimuli))
    for i in range(num_stimuli):
        for j in range(num_stimuli):
            # Compute correlation for trial 0 (before learning) and trial 1 (after learning) separately
            correlation_matrix[i, j] = np.corrcoef(stimuli[i][0], stimuli[j][0])[0, 1]
    return correlation_matrix


def rep_NMPH(v1actm_H, v1actm_L, v1actm_V, v1actm_R):
    # Organize stimuli representations into a list for easier processing
    stimuli = [v1actm_H, v1actm_L, v1actm_V, v1actm_R]

    # Compute correlation matrices for before and after learning
    before_learning = compute_correlation_matrix([s[0:1] for s in stimuli])  # First row for before learning
    after_learning = compute_correlation_matrix([s[1:2] for s in stimuli])  # Second row for after learning

    # Compute the difference matrix (learning effect)
    learning_effect = after_learning - before_learning

    # Reshape matrices for plotting
    before_learning_reshaped = before_learning.reshape(1, -1)
    learning_effect_reshaped = learning_effect.reshape(1, -1)

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.scatter(before_learning_reshaped[0], learning_effect_reshaped[0])
    plt.xlabel('Before Learning (Correlation)')
    plt.ylabel('Learning Effect (Difference in Correlation)')
    plt.title('Learning Curve Based on Neural Representation Similarity')
    plt.grid(True)
    plt.show()

# Example usage (assuming you have the matrices v1actm_H, v1actm_L, v1actm_V, v1actm_R defined)
rep_NMPH(H_act, L_act, V_act, R_act)



