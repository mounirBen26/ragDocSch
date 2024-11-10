from langchain.text_splitter import CharcterTextSplitter
from langchain.document_loader import TextLoder
from langchain.embeddings import HuggingFaceEmbeddings

#load the text data file
loader = TextLoder('./horoscope.txt')
documents = loader.load()
#split the text
text_splitter = CharcterTextSplitter(chunk_size=1000, chunk_overlap=4)
docs = text_splitter.split_documents(documents)
#call embedings
embeddings = HuggingFaceEmbeddings