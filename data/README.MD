#### gloss-annotation.tsv:
This file contains manual annotation of the gloss from PHOENIX 14T dataset.

#### full.train, full.dev, full.test:
These tab separated files contain manual annotation along with tags predicted by our classifier.
The first column denotes whether the label is manual annotation or predicted by the classifier.
The second column contains gloss data of PHOENIX 14T dataset infused with the intensification labels. Gloss tokens that are predicted/annotated to have lower intensity, are surrounded by <LOW-INT> and </LOW-INT>. Tokens with higher intensity are surrounded similarly with <HIGH-INT> and </HIGH-INT>. The tokens that do not have any associated intensity, are not surrounded by anything,
Example:
1. $manual-annotation$	<HIGH-INT> WOLKE </HIGH-INT> LOCH SPEZIELL NORDWEST
- This means the intensity labels for this instance are manually annotated. The token WOLKE has high-intensity, other tokens have no intensity.
2. $tagged$	SAMSTAG AUCH <LOW-INT> FREUNDLICH </LOW-INT>
- This means the intensity labels for this instance are tagged by our classifier. The token FREUNDLICH has low intensity, other tokens have no intensity.
