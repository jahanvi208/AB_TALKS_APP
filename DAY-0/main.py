#today we will clean text and count the frequwncy then


# import re
# import string
# from nltk.corpus import stopwords
# import nltk

# text=input("enter a sentence:")
# text=text.lower()
# text=re.sub(r'\s+\d+',' ',text)
# text=text.translate(str.maketrans('','',string.punctuation))
# nltk.download('stopwords')
# stop_words=set(stopwords.words('english'))
# words=text.split()
# clean_words=[word for word in words if word not in stop_words]
# clean_text=" ".join(clean_words)
# print("clean text is:",clean_text)

# freq={}
# for word in words:
#     if word in freq:
#         freq[word] +=1
    
#     else:
#         freq[word]=1
    
# print("\n words freq")
# print(freq)

# today we will do clean token and bag of wrods
# from nltk.tokenize import word_tokenize
# import string
# from nltk.corpus import stopwords
# import nltk

# text=input("enter a text")
# tokens= word_tokenize(text)
# text=text.translate(str.maketrans('','',string.punctuation))

# nltk.download('stopwords')
# stop_words=