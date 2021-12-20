from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast
from transformers import set_seed
import os
import gradio as gr
import torch

set_seed(42)

model_name = 'gpt2'
max_length = 174

# if not available then run on cpu 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Load a trained model and vocabulary that you have fine-tuned
#configuration = GPT2Config.from_json_file("/root/model_save/config.json")
model = GPT2LMHeadModel.from_pretrained("/root/model_save", return_dict_in_generate=True)
tokenizer = GPT2TokenizerFast.from_pretrained("/root/model_save")

model = model.to(device)

questions_list = [
    "Identity No",
    "Date of Expire",
    "Date of Issue",
    "Date of Birth",
    "Given Name",
    "Surname",
    "Mother’s name",
    "Father’s name",
    "Issuing Authority",
    "----------------"
    "Gender",
    "Document No",
    "Nationality",
    "Greek First Name",
    "Greek Last Name",
    "Mother’s Last Name",
    "Place of Birth",
    "Dimotis",
    "Height", 
    "Tax No",
    "Social Security No",
    "Health No",
    "Mother’s Parents",
    "Father’s Parents"
] 

def gpt2(text, ques):
    prompt = f"{text}, {ques}:\n"

    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    generated_outputs = model.generate(input_ids, 
                                      do_sample=True,   
                                      top_k=50, 
                                      max_length = max_length,
                                      top_p=0.95, 
                                      num_return_sequences=3, 
                                      output_scores=True)


    # only use id's that were generated
    # gen_sequences has shape [3, 15]
    gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:]

    # let's stack the logits generated at each step to a tensor and transform
    # logits to probs
    probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)  # -> shape [3, 15, vocab_size]

    # now we need to collect the probability of the generated token
    # we need to add a dummy dim in the end to make gather work
    gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
    
    # now we can do all kinds of things with the probs

    # the probs that exactly those sequences are generated again
    # those are normally going to be very small
    unique_prob_per_sequence = gen_probs.prod(-1)

    # Get max value of prob out of three
    lst= list(enumerate(unique_prob_per_sequence))
    tp = max(enumerate(unique_prob_per_sequence), key=(lambda x: x[1]))

    #probability
    prob = tp[1].item()

    # output token 
    tok = gen_sequences[tp[0], :, None].squeeze(-1)
    pred = tokenizer.decode(tok, skip_special_tokens=True) 

    return pred, prob


inputs =  [ 
           gr.inputs.Textbox(lines=4, label="Input Text"),
           gr.inputs.Dropdown(questions_list, label="Question") 
]
          

outputs = [ 
           gr.outputs.Textbox(label="GPT-2"),
           gr.outputs.Textbox(label="Score")
]

title = "GPT-2"
description = "Ερωτήσεις για τα Ελληνικά: Identity No, Date of Issue, Greek First Name, Given Name, Greek Last Name, Surname,Father's name,Mother's name,Mothre's Last Name,Date of Birth,Place of Birth, Dimotis \n Ερωτήσεις για τα Ελληνικά: Identity No, Date of Issue, Greek First Name, Given Name, Greek Last Name, Surname,Father's name,Mother's name,Mothre's Last Name,Date of Birth,Place of Birth, Dimotis"
examples = [
    ["Y5015362, 03/08/2018, Τρυφωνία, Truphonia, Καρκανάκη, Karkanake, Θέμης, Φιλαρέτη, Ζαχαρίου, 16/07/1954, Λασσίθι Άρτα, Κόρινθος 34927/4"],
    ["PO4803172, 29/12/2007, Βαλέριος, Balerios, Θεολόγος, Theologos, Αχιλλέας, Παντούλα, Παυλή, 13/08/1980, Χίος Πολύγυρος, Κόρινθος 82397/5"],
    ["P7752142, 21/03/2007, Δράκων, Drakon, Θεοδωρικάκος, Theodorikakos, Αριστοφάνης, Ροδόκλεια, Τζιόβα, 15/02/1923, Θεσπρωτία Καρπενήσι, Λιβαδιά 96933/9"],
    ["I6641241, 06/03/2016, Μαρκέλλα, Markella, Κωτσιονοπούλου, Kotsionopoulou, Ιάσονας, Κλαίρη, Κοντού, 10/07/1972, Πέλλα Χίος, Ηγουμενίτσα 57776/4"],
    ["TH3925465, 31/08/2019, Γραμματική, Grammatike, Κάκκα, Kakka, Λεμονής, Μιχαέλα, Ταφραλή, 29/04/1935, Αργολίδα Έδεσσα, Λάρισα 31165/2"],
    ["AM2456789, 02/08/2018, Έριον, Erion, Τσάνι, Tsani, Μπέντρι, Ζελιχά, Νέτα, 03/02/1995, Λούσνιε, Αλβανίας Λεβαδία 27097/1"],
    ["XY5338112, 01/08/2015, Στεργιανή, Stergiane, Καρακώστα, Karakosta, Μάριος, Ρωξάνη, Καλαμάρα, 12/02/1974, Λέσβος Ναύπλιο, Λιβαδιά 41874/6"],
    ["XE5338112, 06/07/2016, Ευθύμιος, Efthymios, Τσέργας, Tsergas, Κωσταντίνος, Πηνελόπη, Κάσσου, 26/07/1987, Αθήνα Αττικής, Δημητρίου 15257/6"],
    ["Azevedo, Íris, F, 1.7, PRT, 05/12/1962, 71384960, 71384960 7 ZZ5, 29/06/2028, 7624028, 670359911983819, 684983982007, Duarte Brito, Joana Cruz"],
    ["Garcia, Frederico, M, 2.01, PRT, 22/06/1999, 33473483, 33473483 2 ZZ9, 11/12/2021, 8805025, 341937193817616 749286958000 Valentim Azevedo Adriana Tavares"],
    ["Pinho, Maria, F, 1.81, ZAF, 03/10/1943, 52346139, 52346139 8 ZZ4, 28/06/2031, 7675668, 287316863051348, 656175918929, Tomé Rocha, Kelly Garcia"],
    ["Ramos, Júlia, F, 1.74, PRT, 14/04/1958, 65787015, 65787015 0 ZZ4, 27/07/2024, 3677651, 957790967551072, 616748192419 Micael Moura, Catarina Pinheiro"],
    ["Mota, Francisca, F, 1.38, PRT, 01/01/1987, 61839423, 61839423 7 ZZ1, 30/01/2031, 8302449, 565298651057420, 425556485877, William Silva, Catarina Ribeiro"],
    ["1839028408, AKÇAY, ADAK, 21/04/1969, E/M, A23T95764, TC/TUR, 20/09/2027, SEVGINUR, İMAM"],
    ["24744254287, İHSANOĞLU, SÜNER, 30/04/1942, K/F, X13M12645, TC/TUR, 10/11/2023, REHIME, MEMILI"],
    ["84695983927, SEZGIN, NILI, 29/10/1914, K/F, O41O18520, DOM, 13/02/2024, ŞAHINDER, SUDI"],
    ["62872738140, SEZER, EMINE, 26/05/1981, K/F, A87T96806, TC/TUR, 04/03/2031, SERNUR, MÜZEKKER"]
]

gr.Interface(gpt2, inputs, outputs, title=title, description=description, examples=examples).launch(share=True)

print("ok!")
