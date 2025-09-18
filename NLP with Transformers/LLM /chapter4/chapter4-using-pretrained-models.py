from transformers import pipeline

camembert_fill_mask = pipeline("fill-mask", model = "camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")

