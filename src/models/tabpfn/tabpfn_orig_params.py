# https://github.com/automl/TabPFN/issues/26#issuecomment-1407630200

import json

from tabpfn import TabPFNClassifier

classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

# Save dict as json
# Sort keys to make it easier to compare
json.dump(classifier.c, open("tabpfn_orig_params.json", "w"), indent=4, sort_keys=True)

