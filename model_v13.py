#!/usr/bin/env python
# coding: utf-8

# In[1]:

#running
# 1/2 hr per epoch on 300k records
# python3 model_v11.py
# REQUIREMENTS  
# pip3 install tensorflow-gpu==2.0.0-beta1

#preprocessing
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.ribes_score import corpus_ribes
import warnings
warnings.filterwarnings("ignore")


import os


os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



#tensorflow setup
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import io
import time
import keras
import jieba
import re


#extract zh tokens
def extract_zh_tokens(doc):
    word = []
    for sent in doc.sentences:
        for wrd in sent.words:
            #extract text and lemma
            word.append(wrd.text)
    return word



def read_file(path):
    en_list = []
    en_token_list = []
    zh_list = []
    zh_token_list = []
    en_sentence_full= []
    zh_sentence_full = []
    zh_sentence_tokenized_full = []
    en_sentence_tokenized_full = []
    zh_token_list_sentence_no_tags = []
    en_token_list_sentence_no_tags = []
    idx = 0
    with open(path,encoding = "utf-8", errors='ignore') as f:
        line_list = []
        for line in f:              #print the list items

            try:

                line = re.sub(r"(?<=[A-Za-z]{1})\t(?=[A-Za-z\“\,]+)"," ",line)
                line = re.sub(r"([\:\?\.\!\,\¿\。\！\”\“\`\/\(\)-]\？)", r"", line)
                line = re.sub(r"\.", r"", line)
                line = re.sub(r"\?", r"", line)
                line = re.sub(r"\？", r"", line)
                line = re.sub(r"\。", r"", line)

                line = re.sub(r"\xa0", r"", line)
                line = re.sub(r"\~", r"", line)
                line = re.sub(r"\（", r"", line)
                line = re.sub(r"\）", r"", line)
                line = re.sub(r"\·", r"", line)
                line = re.sub(r"\–", r"", line)
                line = re.sub(r"\-", r"", line)
                line = re.sub(r"\•", r"", line)
                line = re.sub(r"\、", r"", line)
                line = re.sub(r"\，", r"", line)
                line = re.sub(r"\，", r"", line)
                line = re.sub(r"\"", r"", line)
                line = re.sub(r"\'", r"", line)
                line = re.sub(r"\’", r"", line)

                line = re.sub(r"\！", r"", line)
                line = re.sub(r"\；", r"", line)
                line = re.sub(r"\：", r"", line)
                line = re.sub(r"\［", r"", line)
                line = re.sub(r"\］", r"", line)
                line = re.sub(r"\【", r"", line)
                line = re.sub(r"\】", r"", line)
                line = re.sub(r"\—", r"", line)
                line = re.sub(r"—", r"", line)

                line = re.sub(r"\n", r"", line)
                line = re.sub(r'\d+', '', line)

                en,zh = line.split('\t')

                if len(en) > 1: # remove single character english lines
               
                  zh_token_list = jieba.cut(zh , cut_all=False)

                  sequence_len = 10





                  zh_token_list_2 =  " ".join(zh_token_list)

                  zh_token_list_2 = zh_token_list_2.split(" ")
                  zh_token_list_2 = zh_token_list_2[0:sequence_len] # take only first 10 words
                  zh_token_list_2 =  " ".join(zh_token_list_2)

                  
                  zh_token_list_sentence = "<start> " +  zh_token_list_2 +" <end>"


                  zh_word_list = zh_token_list_2.split(" ")
                  # print("zh tokens: ", zh_word_list) # tokenized chinese words

                  for i in range(len(zh_word_list)):
                      zh_word_list[i] = "<start> " + zh_word_list[i] +" <end>"
                  

                  en_token_list =  nltk.word_tokenize(en)
                  en_word_list = []
                  for i in range(len(en_token_list)):
                      en_word_list.append("<start> " + en_token_list[i] +" <end>")
          
                  


                  en_token_list_2 =  " ".join(en_token_list) 

                  en_token_list_2 = en_token_list_2.split(" ")
                  en_token_list_2 = en_token_list_2[0:sequence_len] # take only first 10 words
                  en_token_list_2 =  " ".join(en_token_list_2)


                  en_token_list = "<start> "+en_token_list_2 +" <end>"


                  # print("zh_token_list_sentence: ", zh_token_list_sentence)
                  if (en_token_list != "<start>  <end>") & (zh_token_list_sentence != "<start>  <end>"): # remove blank records
                  
                    zh_sentence_full.append(zh_token_list_sentence)
                    en_sentence_full.append(en_token_list)
                    
                    en_token_list_sentence_no_tags.append(en_token_list_2)    
                    zh_token_list_sentence_no_tags.append(zh_token_list_2)


                    zh_sentence_tokenized_full.append(zh_word_list)
                    en_sentence_tokenized_full.append(en_word_list)


            except ValueError:
                continue
                
            idx +=1
            # if idx == 2500:
            if idx == 5000:
                break
                
    return en_sentence_full,zh_sentence_full, en_sentence_tokenized_full,zh_sentence_tokenized_full,zh_token_list_sentence_no_tags,en_token_list_sentence_no_tags

print("processing sentences")
filename = 'data.stat.org_news-comm_v14_trai_news-comm-_JzD8orWKHLY7s4d_7EUj5crr2YhvIe6BJo64Kq0Als'

en_sentence_full,zh_sentence_full, en_sentence_tokenized_full,zh_sentence_tokenized_full,zh_token_list_sentence_no_tags,en_token_list_sentence_no_tags = read_file(filename)

def max_length(tensor):
    return max(len(t) for t in tensor)

df = pd.DataFrame( {'en_sentence_full': en_sentence_full,
     'en_sentence_tokenized_full': en_sentence_tokenized_full,
    'zh_sentence_full': zh_sentence_full,
                    'zh_sentence_tokenized_full': zh_sentence_tokenized_full,
                    'zh_token_list_sentence_no_tags' :zh_token_list_sentence_no_tags,
                    'en_token_list_sentence_no_tags' :en_token_list_sentence_no_tags
    })

# dropping ALL duplicte values 
df.drop_duplicates(subset ="en_sentence_full",  keep = False, inplace = True) 

df_test_input = df[['en_token_list_sentence_no_tags']]
df_test_output = df[['zh_token_list_sentence_no_tags']]


df['length'] = df['en_sentence_full'].str.len()
df.sort_values('length', ascending=True, inplace=True)


print("processed dataframe shape: ", df.shape)

EPOCHS = 1 # anything greater than 22 on a full run seems to crash on K80 Azure
# EPOCHS = 10 # anything greater than 10 on a full run seems to crash on 2070
# df = df.head(90000)


# # min character length is 15
en_sentence_full = df['en_sentence_full'].tolist()
zh_sentence_full = df['zh_sentence_full'].tolist()

print(en_sentence_full[0:20])
print(zh_sentence_full[0:20])

#text tokenizer to indexed sequences padded 
def tokenize(text):
  text_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  text_tokenizer.fit_on_texts(text)
  tensor = text_tokenizer.texts_to_sequences(text)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
  return tensor, text_tokenizer


# function to create maps between index and word
def convert(text, tensor):
  for item in tensor:
    if item != 0:
      print("{} => {}".format(item, text.index_word[item]))

print("tokenizing text")
input_tensor, inp_text_tokenizer = tokenize(en_sentence_full)
target_tensor, targ_text_tokenizer = tokenize(zh_sentence_full)

print('english index tensor\n')
print(input_tensor[0:20])
print ("Input (English) Language - index to word mapping @ index 0")
convert(inp_text_tokenizer, input_tensor[0])

print('chinese index tensor\n')
print(target_tensor[0:20])
print ("Target (Chinese) Language - index to word mapping @ index 0")
convert(targ_text_tokenizer, target_tensor[0])
convert(targ_text_tokenizer, target_tensor[9])


max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)



print ()



print("splitting into training and testing")
train_test_split_ratio = 0.0016
random_seed_state = 52

#train test splitting for tensors
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=train_test_split_ratio,random_state=random_seed_state)


# #split original sentences, not tensors, for performance evaluation
input_train, input_val, target_train, target_val = train_test_split(df_test_input, df_test_output, test_size=train_test_split_ratio,random_state=random_seed_state)

input_val = input_val.values
target_val = target_val.values


print('input_tensor_train.shape' + str(input_tensor_train.shape))
print('target_tensor_train.shape' + str(target_tensor_train.shape))
print('input_tensor_val.shape' + str(input_tensor_val.shape))
print('target_tensor_val.shape' + str(target_tensor_val.shape))


# reduce batch size for memory
buffer_size = len(input_tensor_train)
batch_size = 128
embedding_dim = 256
units = 1024

steps_per_epoch = len(input_tensor_train)//batch_size

vocab_input_size = len(inp_text_tokenizer.word_index) + 1
vocab_target_size = len(targ_text_tokenizer.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(buffer_size)
dataset = dataset.batch(batch_size, drop_remainder=True)

print('setting up network')

# with tf.device("/gpu:0"):
class encoder_block(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, encoder_units, batch_sz):
    super(encoder_block, self).__init__()
    self.batch_sz = batch_sz
    self.encoder_units = encoder_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.encoder_units, return_sequences=True,recurrent_initializer='glorot_uniform',return_state=True),merge_mode ='concat')
    self.gru_old = tf.keras.layers.GRU(self.encoder_units, return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)

    output, forward_h, forward_c, backward_h, backward_c = self.lstm(x, initial_state=hidden)
    # output, state = self.gru_old(x, initial_state=hidden)
    # state_h = tf.keras.layers.Concatenate()([forward_h, backward_h]) # hidden state
    state_h = tf.keras.layers.Average()([forward_h, backward_h]) # hidden state
    # state_c = tf.keras.layers.Concatenate()([forward_c, backward_c]) # cell statee

    print('state_h: ', state_h)
    # print('state: ', state)
    return output, forward_h

  def initialize_hidden_state(self):
    # return tf.zeros((self.batch_sz, self.encoder_units)) #[tf.zeros((self.batch_sz, self.encoder_units)),tf.zeros((self.batch_sz, self.encoder_units))]
    init_state = [tf.zeros((self.batch_sz, self.encoder_units)) for i in range(4)]
    return init_state

encoder = encoder_block(vocab_input_size, embedding_dim, units, batch_size)

class bahdanau_attention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(bahdanau_attention, self).__init__()
    self.weights_1 = tf.keras.layers.Dense(units)
    self.weights_2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    hidden_w_t_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh( self.weights_1(values) + self.weights_2(hidden_w_t_axis)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vec = attention_weights * values
    context_vec = tf.reduce_sum(context_vec, axis=1)

    return context_vec, attention_weights

attention_layer = bahdanau_attention(11)

class decoder_block(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, decoder_units, batch_sz):
    super(decoder_block, self).__init__()
    self.batch_sz = batch_sz
    self.decoder_units = decoder_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.decoder_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    self.full_connected = tf.keras.layers.Dense(vocab_size)
    self.attention = bahdanau_attention(self.decoder_units) # attention layer

  def call(self, x, hidden, encoder_output):
    context_vec, attention_weights = self.attention(hidden, encoder_output)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vec, 1), x], axis=-1)
    output, state = self.gru(x)
    output = tf.reshape(output, (-1, output.shape[2]))
    x = self.full_connected(output)
    return x, state, attention_weights

decoder = decoder_block(vocab_target_size, embedding_dim, units, batch_size)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "checkpoint")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)


# In[ ]:


@tf.function
def train_step(input, target, encoder_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    encoder_output, encoder_hidden = encoder(input, encoder_hidden)
    decoder_hidden = encoder_hidden

    decoder_input = tf.expand_dims([targ_text_tokenizer.word_index['<start>']] * batch_size, 1)
    
    # Teacher forcing
    for t in range(1, target.shape[1]):

      predictions, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
      loss += loss_function(target[:, t], predictions)

      # using teacher forcing
      decoder_input = tf.expand_dims(target[:, t], 1)
  batch_loss = (loss/int(target.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss


for epoch in range(EPOCHS):
  start = time.time()

  encoder_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  print('training')

  for (batch, (input,target)) in enumerate(dataset.take( steps_per_epoch)):

    #increase RAM size during training
    batch_loss = train_step(input, target, encoder_hidden)
    total_loss += batch_loss

    if batch % 100 == 0:
        print('Epoch: {} Batch: {} Loss: {:.4f}'.format(epoch + 1,batch, batch_loss.numpy()))

  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss/steps_per_epoch))
  print('Epoch duration {} s\n'.format(time.time() - start))



#translation functions: (prediction)

def translate_preprocess(line):
    en_list = []
    en_token_list = []
    zh_list = []
    zh_token_list = []
    en_sentence_full= []
    zh_sentence_full = []
    zh_sentence_tokenized_full = []
    en_sentence_tokenized_full = []
    idx = 0

    line = re.sub(r"(?<=[A-Za-z]{1})\t(?=[A-Za-z\“\,]+)"," ",line)
    line = re.sub(r"([?.!,¿])", r"", line)
    line = re.sub(r"\n", r"", line)

    en_token_list =  nltk.word_tokenize(line)
    en_word_list = []
    for i in range(len(en_token_list)):
        en_word_list.append("<start> " + en_token_list[i] +" <end>")
        
    en_token_list =  " ".join(en_token_list)        
    en_token_list = "<start> "+en_token_list +" <end>"

    en_sentence_tokenized_full.append(en_word_list)
    
    
    en_sentence_full.append(en_token_list)

    return "".join(en_sentence_full)



def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = translate_preprocess(sentence)

    inputs = [inp_text_tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units)) for i in range(4)]
    # hidden = [tf.zeros((1, units))]

    encoder_output, encoder_hidden = encoder(inputs, hidden)

    decoder_hidden = encoder_hidden
    decoder_input = tf.expand_dims([targ_text_tokenizer.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_output)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_text_tokenizer.index_word[predicted_id] + ' '

        if targ_text_tokenizer.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        decoder_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    # print('Input: %s' % (sentence))
    # print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    # plot_attention(attention_plot, sentence.split(' '), result.split(' '))

    return result



  # BLEU -> RIBES is better

  
from nltk.translate.bleu_score import sentence_bleu


# input_val
print('******************')
print(input_val[0:5])
# # , target_val 
print('******************')
print(target_val[0:5])


print("--------------------------------------------")
original_translations = []
predicted_translations = []

prediction_tokens_full = []
target_tokens_full = []
remaining_input_val = []
score_sent_list = []
score_ribes_sent_list = []
indxj = 0
for i in range(len(input_val)):
  print('input_val.shape: ', input_val.shape)
  print(i)
  try:
    print('english, ' " ".join(str(x) for x in input_val[i]))

    prediction = translate(" ".join(str(x).lower() for x in input_val[i]))
    print('prediction: ',prediction)

    # print('prediction: ', prediction)
    prediction_token_list = jieba.cut(prediction , cut_all=False)
    prediction_token_list =  " ".join(prediction_token_list)

    prediction_token_list = prediction_token_list.split(" ")
    # print(prediction_token_list)



    # print('target_val[i]): ',target_val[i])

    #
    # target_token_list = jieba.cut(target , cut_all=False)
    # target_token_list =  " ".join(target_token_list)
    target = " ".join(str(x) for x in target_val[i])
    target_token_list = target.split(" ")
    # print('target_token_list): ',target_token_list)
    # print(target_token_list)

    # perform removal 
    
    while("" in prediction_token_list) : 
      prediction_token_list.remove("") 

    if '<' in prediction_token_list:
      prediction_token_list.remove('<')
      prediction_token_list.remove('end')
      prediction_token_list.remove('>')


    score_sent = sentence_bleu(list(target_token_list), prediction_token_list)

    print('bleu sent score: ' + str(score_sent))

    print([[target_token_list]])
    print([prediction_token_list])

    try:
      ribes_score = round(corpus_ribes([[target_token_list]], [prediction_token_list]),4)
    except ZeroDivisionError:
      ribes_score = 0
    print('ribes sent score: ' + str(ribes_score))

    prediction_tokens_full.append(prediction_token_list)
    target_tokens_full.append(target_token_list)
    remaining_input_val.append(np.array2string(input_val[i]))
    score_sent_list.append(score_sent)
    score_ribes_sent_list.append(ribes_score)


  except KeyError:
      continue 

  # if indxj == 50:
  #   break

  indxj+=1

# print('final bleu sent score list = ' + str(score_sent_list))
print('final bleu sent score average = ' + str(sum(score_sent_list)/len(score_sent_list)))
# print('final ribes sent score list = ' + str(score_ribes_sent_list))
print('final ribes sent score average = ' + str(sum(score_ribes_sent_list)/len(score_ribes_sent_list)))

print("--------------------------------------------")


target_tokens_full = list(target_tokens_full)


print("88888888888888888888888888888888888888888888")
# print(target_tokens_full)
# print(len(target_tokens_full))
# print(prediction_tokens_full)
# print(len(prediction_tokens_full))

# # reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
# # candidate = ['this', 'is', 'a', 'test']


# print("English")
# print(remaining_input_val)
# print("Ground Truth")
# print(target_tokens_full)
# print("Predictions")
# print(prediction_tokens_full)
# print(len(remaining_input_val))


for i in range(len(remaining_input_val)):
  print("set : ", i)
  print("english:\t\t", remaining_input_val[i])
  print("chinese:\t\t", target_tokens_full[i])
  print("chinese pred:\t\t", prediction_tokens_full[i])
  print()

dict_output = {'english':remaining_input_val,
              'chinese':target_tokens_full,
              'chinese_pred':prediction_tokens_full,
              'bleu':score_sent_list,
              'ribes':score_ribes_sent_list}

df_output = pd.DataFrame(dict_output)


# df_output.to_csv('model_v12_epochs_' +str(EPOCHS)+'_avg_bleu_' + str(round(sum(score_sent_list)/len(score_sent_list),5))+ '_ribes_'+ str(round(sum(score_ribes_sent_list)/len(score_ribes_sent_list),5)) +'.csv')
df_output.to_csv('model_v12_epochs_' +str(EPOCHS)+'_avg_bleu_' + str(sum(score_sent_list)/len(score_sent_list))+ '_ribes_'+ str(sum(score_ribes_sent_list)/len(score_ribes_sent_list)) +'.csv')

# # score = sentence_bleu(reference, candidate)
# score = corpus_bleu(target_tokens_full, prediction_tokens_full)
# print('Bleu')
# print(score)


# #reshape target tokens for ribes
# target_tokens_full = [target_tokens_full]
# print("Ground Truth Ribes")
# print(target_tokens_full)

# prediction_tokens_full = prediction_tokens_full
# print("Predictions Ribes")
# print(prediction_tokens_full)


# # target_tokens_full = [[['东京', '日本首相', '安倍晋三', '在', '年', '执政', '后', '不久', '就', '开始']]]
# # prediction_tokens_full = [['东京', '日本首相', '安倍晋三', '在', '年', '执政', '后', '不久', '就', '开始']]
# # target_tokens_full = [[['我', '喜欢', '自然', '语言', '处理'],['我', '喜欢', '自然', '语言', '处理']]]
# # prediction_tokens_full = [['我', '喜欢', '自然', '语言', '处理'],['我', '喜欢', '自然', '语言', '处理']]



# ribes_score = round(corpus_ribes(target_tokens_full, prediction_tokens_full),4)
# print('Ribes')
# print(ribes_score)