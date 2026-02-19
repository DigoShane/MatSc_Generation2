#This piece of code is there to be run after we train the tokenizer.
#This will validate the tokenizer.

from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

text = "YBa2Cu3O7 exhibits superconductivity at Tc ≈ 92K."

#Runs the full tokenization pipeline on the input text
output = tokenizer.encode(text)

print(output.tokens)
#o/p was -> ['ĠYBa', '2', 'Cu', '3', 'O', '7', 'Ġexhibits', 'Ġsuperconductivity', 'Ġat', 'ĠTc', 'ĠâīĪ', 'Ġ92', 'K', '.']
#Ġ -> special byte-level marker for word start

