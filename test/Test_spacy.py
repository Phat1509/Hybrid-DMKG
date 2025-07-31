from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import spacy


head = "Central Time Zone"
nlp = spacy.load("../en_core_web_md")
nlp.add_pipe("entityLinker", last=True)
pattern = r'Q\d+'
head_linking = re.search(pattern, str(nlp(head)._.linkedEntities))
# tail_linking = re.search(pattern, str(nlp(tail)._.linkedEntities))
entity = nlp.get_entity("Q329")
print(entity)
print("1111111111"*30)
print(head_linking)
