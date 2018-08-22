# -*- coding: utf-8 -*-

import string
import numpy as np
import math

def hash_string(s):

    return abs(hash(s)) % (10 ** 8)

def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt(sum(word_vectors[word]**2) + 1e-6)
        word_vectors[word] *= norm
    return word_vectors

def xavier_vector(word, D=300):
    """
    Returns a D-dimensional vector for the word. 

    We hash the word to always get the same vector for the given word. 
    """

    seed_value = hash_string(word)
    np.random.seed(seed_value)

    neg_value = - math.sqrt(6)/math.sqrt(D)
    pos_value = math.sqrt(6)/math.sqrt(D)

    rsample = np.random.uniform(low=neg_value, high=pos_value, size=(D,))
    norm = np.linalg.norm(rsample)
    rsample_normed = rsample/norm

    return rsample_normed

def process_dialogues(data, ontology, max_no_turns=-1):

    for name, dialogue in data.items():
        
        keylist = list(range(dialogue['len']))
        no_turns = len(keylist)
        if no_turns > max_no_turns:
            max_no_turns = no_turns
        for i, key in enumerate(keylist):
            turn = dialogue[str(key)]
            belief_state = turn['user']['belief_state']
            for domain in belief_state:
                slots = belief_state[domain]['semi']
                bookings = belief_state[domain]['book']
                for booking in bookings:
                    if booking != "booked" and bookings[booking] != "":
                        slots["book " + booking] = bookings[booking]
                        
                new_slots= {}
                for slot in slots:
                    value = slots[slot]
                    slot, value = clean_domain(domain, slot, value)
                    new_slots[slot] = value
                    assert value != "not mentioned"
                    
                    if value != "":
                        key = domain+"-"+slot
                        if key in ontology:
                            if value not in ontology[key]:
                                ontology[key].append(value)
                        else:
                            ontology[key] = []
                belief_state[domain]['semi'] = new_slots
    return max_no_turns

def clean_text(text):
    text = text.strip()
    text = text.lower()
    text = text.replace(u"’", "'")
    text = text.replace(u"‘", "'")
    text = text.replace("don't", "do n't")
    return text    
   
def clean_domain(domain, slot, value):
    value = clean_text(value)
    if not value:
        value = ''
    elif value == 'not mentioned':
        value = ''
    elif domain == 'attraction':
        if slot == 'name':
            if value == 't':
                value = ''
            if value=='trinity':
                value = 'trinity college'
        elif slot == 'area':
            if value in ['town centre', 'cent', 'center', 'ce']:
                value = 'centre'
            elif value in ['ely', 'in town', 'museum', 'norwich', 'same area as hotel']:
                value = ""
            elif value in ['we']:
                value = "west"
        elif slot == 'type':
            if value in ['m', 'mus', 'musuem']:
                value = 'museum'
            elif value in ['art', 'architectural']:
                value = "architecture"
            elif value in ['churches']:
                value = "church"
            elif value in ['coll']:
                value = "college"
            elif value in ['concert', 'concerthall']:
                value = 'concert hall'
            elif value in ['night club']:
                value = 'nightclub'
            elif value in ['mutiple sports', 'mutliple sports', 'sports', 'galleria']:
                value = 'multiple sports'
            elif value in ['ol', 'science', 'gastropub', 'la raza']:
                value = ''
            elif value in ['swimmingpool', 'pool']:
                value = 'swimming pool'
            elif value in ['fun']:
                value = 'entertainment'   
            
    elif domain == 'hotel':
        if slot == 'area':
            if value in ['cen', 'centre of town', 'near city center', 'center']:
                value = 'centre'
            elif value in ['east area', 'east side']:
                value = 'east'
            elif value in ['in the north', 'north part of town']:
                value = 'north'
            elif value in ['we']:
                value = "west"
        elif slot == "book day":
            if value == "monda":
                value = "monday"
            elif value == "t":
                value = "tuesday"
        elif slot == 'name':
            if value == 'uni':
                value = 'university arms hotel'
            elif value == 'university arms':
                value = 'university arms hotel'
            elif value == 'acron':
                value = 'acorn guest house'
            elif value == 'ashley':
                value = 'ashley hotel'
            elif value == 'arbury lodge guesthouse':
                value = 'arbury lodge guest house'
            elif value == 'la':
                value = 'la margherit'
            elif value == 'no':
                value = ''
        elif slot == 'internet':
            if value == 'does not':
                value = 'no'
            elif value in ['y', 'free', 'free internet']:
                value = 'yes'
            elif value in ['4']:
                value = ''
        elif slot == 'parking':
            if value == 'n':
                value = 'no'
            elif value in ['free parking']:
                value = 'free'
            elif value in ['y']:
                value = 'yes'
        elif slot in ['pricerange', 'price range']:
            slot = 'price range'
            if value == 'moderately':
                value = 'moderate'
            elif value in ['any']:
                value = "do n't care"
            elif value in ['any']:
                value = "do n't care"
            elif value in ['inexpensive']:
                value = "cheap"
            elif value in ['2', '4']:
                value = ''
        elif slot == 'stars':
            if value == 'two':
                value = '2'
            elif value == 'three':
                value = '3'
            elif value in ['4-star', '4 stars', '4 star', 'four star', 'four stars']:
                value= '4'
        elif slot == 'type':
            if value == '0 star rarting':
                value = ''
            elif value == 'guesthouse':
                value = 'guest house'
            elif value not in ['hotel', 'guest house', "do n't care"]:
                value = ''
    elif domain == 'restaurant':
        if slot == "area":
            if value in ["center", 'scentre', "center of town", "city center", "cb30aq", "town center", 'centre of cambridge', 'city centre']:
                value = "centre"
            elif value == "west part of town":
                value = "west"
            elif value == "n":
                value = "north"
            elif value in ['the south']:
                value = 'south'
            elif value not in ['centre', 'south', "do n't care", 'west', 'east', 'north']:
                value = ''
        elif slot == "book day":
            if value == "monda":
                value = "monday"
            elif value == "t":
                value = "tuesday"
        elif slot in ['pricerange', 'price range']:
            slot = 'price range'
            if value in ['moderately', 'mode', 'mo']:
                value = 'moderate'
            elif value in ['not']:
                value = ''
            elif value in ['inexpensive', 'ch']:
                value = "cheap"
        elif slot == "food": 
            if value == "barbecue":
                value = "barbeque"
        elif slot == "pricerange":
            slot = "price range"
            if value == "moderately":
                value = "moderate"
        elif slot == "book time":
            if value == "9:00":
                value = "09:00"
            elif value == "9:45":
                value = "09:45"
            elif value == "1330":
                value = "13:30"
            elif value == "1430":
                value = "14:30"
            elif value == "9:15":
                value = "09:15"
            elif value == "9:30":
                value = "09:30"
            elif value == "1830":
                value = "18:30"
            elif value == "9":
                value = "09:00"
            elif value == "2:00":
                value = "14:00"
            elif value == "1:00":
                value = "13:00"
            elif value == "3:00":
                value = "15:00"
    elif domain == 'taxi':
        if slot in ['arriveBy', 'arrive by']:
            slot = 'arrive by'
            if value == '1530':
                value = '15:30'
            elif value == '15 minutes':
                value = ''
        elif slot in ['leaveAt', 'leave at']:
            slot = 'leave at'
            if value == '1:00':
                value = '01:00'
            elif value == '21:4':
                value = '21:04'
            elif value == '4:15':
                value = '04:15'
            elif value == '5:45':
                value = '05:45'
            elif value == '0700':
                value = '07:00'
            elif value == '4:45':
                value = '04:45'
            elif value == '8:30':
                value = '08:30'
            elif value == '9:30':
                value = '09:30'
            value = value.replace(".", ":")
        
    elif domain == 'train':
        if slot in ['arriveBy', 'arrive by']:
            slot = 'arrive by'
            if value == '1':
                value = '01:00'
            elif value in ['does not care', 'doesnt care', "doesn't care"]:
                value = "do n't care"
            elif value == '8:30':
                value = '08:30'
            elif value == 'not 15:45':
                value = ''
            value = value.replace(".", ":")
        elif slot == 'day':
            if value =='doesnt care' or value == "doesn't care":
                value = "do n't care"
        elif slot in ['leaveAt', 'leave at']:
            slot = 'leave at'
            if value == '2:30':
                value = '02:30'
            elif value == '7:54':
                value = '07:54'
            elif value == 'after 5:45 pm':
                value = '17:45'
            elif value in ['early evening', 'friday', 'sunday', 'tuesday', 'afternoon']:
                value = ''
            elif value == '12':
                value = '12:00'
            elif value == '1030':
                value = '10:30'
            elif value == '1700':
                value = '17:00'
            elif value in ['does not care', 'doesnt care', 'do nt care', "doesn't care"]:
                value = "do n't care"
                
            value = value.replace(".", ":")
    if value in ['dont care', "don't care", "do nt care", "doesn't care"]:
        value = "do n't care"

    return slot, value
