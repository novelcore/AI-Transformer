from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from transformers import set_seed
import os
import gradio as gr
import torch

set_seed(42)

model_name = 'gpt2'
max_length = 242

# if not available then run on cpu 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Load a trained model and vocabulary that you have fine-tuned
configuration = GPT2Config.from_json_file("/root/model_save/config.json")
model = GPT2LMHeadModel.from_pretrained("/root/model_save", config=configuration)
tokenizer = GPT2Tokenizer.from_pretrained("/root/model_save")

model = model.to(device)

questions_list = [
                  "Αριθμός Ταυτότητας", 
                  "Ημερονηνία Έκδοσης", 
                  "Όνομα", 
                  "Given Name", 
                  "Επώνυμο", 
                  "Surname",
                  "Πατρώνυμο",
                  "Μητέρας Όνομα",
                  "Επίθετο Μητέρας",
                  "Ημερονηνία Γέννησης",
                  "Καταγωγή", 
                  "Δημότης", 
                  "Αρχή Έκδοσης Δελτίου"
]

def gpt2(text, ques):
    prompt = f"{text}, {ques}:"
    model.eval()

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    input_len = len(generated[0])
    generated = generated.to(device)


    sample_output = model.generate(
                                generated, 
                                do_sample=True,   
                                top_k=50, 
                                max_length = max_length,
                                top_p=0.95, 
                                num_return_sequences=1
                                )

    return tokenizer.decode(sample_output[0][input_len:], skip_special_tokens=True)


inputs =  [ 
           gr.inputs.Textbox(lines=4, label="Input Text"),
           gr.inputs.Dropdown(questions_list, label="Ερώτηση") 
]
          

outputs =  gr.outputs.Textbox(label="GPT-2")

title = "GPT-2"
description = "demo for OpenAI GPT-2. To use it, simply add your text, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://openai.com/blog/better-language-models/'>Better Language Models and Their Implications</a> | <a href='https://github.com/openai/gpt-2'>Github Repo</a></p>"
examples = [
            ["ZB7217457 20/01/2007 Περικλής Perikles Σταυρόπουλος Stauropoulos Ταξιάρχης Χριστίνα Τριφτανίδου 05/06/1954 Χαλκιδική Λευκάδα Αγ. Νικόλαος 29967/2 Υ.Α. Λευκάδα Χαλκιδική"],
            ["A0391706 06/09/2015 Αίας Aias Μαυρουδής Mauroudes Βλάσης Συμέλα Τουλάκη 16/01/1949 Κέρκυρα Πολύγυρος Κοζάνη 40319/2 Υ.Α. Πολύγυρος Κέρκυρα"],
            ["P1611587 22/11/2014 Φιλαρέτη Philarete Τσιωλξ Tsiolx Παντελεήμων Βρυσηίς Τσετσέρη 03/10/1942 Καρδίτσα Κέρκυρα Λευκάδα 89330/4 Υ.Α. Κέρκυρα Καρδίτσα"],
            ["K7054094 16/08/2017 Αχιλλέας Akhilleas Μπερεδήμας Mperedemas Ιωσήφ Αθηνά Κωτσιονοπούλου 03/11/1951 Φθιώτιδα Ξάνθη Πάτρα 59412/8 Υ.Α. Ξάνθη Φθιώτιδα"],
            ["MA2982052 06/04/2011 Βελισσαρία Belissaria Κόλλια Kollia Ασημάκης Θεολογία Μαρτιάδου 21/09/1925 Καστοριά Ναύπλιο Χαλκίδα 12343/7 Υ.Α. Ναύπλιο Καστοριά"],
            ["AM2456789 02/08/2018 Έριον Erion Τσάνι Tsani Μπέντρι Ζελιχα Ζετζα 03/02/1995 Λούσνιε Αλβανίας Λεβαδέων 27097/1 Υ.Α. Λιβαδειά Θήβα"],
            ["XY5338112 01/08/2015 Στεργιανή Stergiane Καρακώστα Karakosta Μάριος Ρωξάνη Καλαμάρα 12/02/1974 Λέσβος Ναύπλιο Λιβαδιά 41874/6 Υ.Α. Ναύπλιο Λέσβος"],
            ["XE5338112 06/07/2016 Ευθύμιος Efthymios Τσέργας Tsergas Κωσταντίνος Πηνελόπη Κάσσου 26/07/1987 Αθήνα Αττικής Δημητρίου 15257/6 Υ.Α. Αγίου Δημητρίου"]
]

gr.Interface(gpt2, inputs, outputs, title=title, description=description, article=article, examples=examples).launch(share=True)

print("ok!")
