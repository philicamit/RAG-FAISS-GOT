import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

load_dotenv()

# folder_path = "GOT_Book"
# output_file = "combined_got_books.txt"

# # Read and combine all text files in the folder into a single output file

# with open(output_file, "w", encoding="utf-8") as outfile:
#     for file_name in sorted(os.listdir(folder_path)):
#         if file_name.endswith(".txt"):
#             file_path = os.path.join(folder_path, file_name)
#             outfile.write(f"\n\n=== {file_name} ===\n\n")

#                 # errors="ignore" invalid bytes
#             with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
#                 outfile.write(infile.read())

# # Loader
# loader = TextLoader(output_file, encoding="utf-8")
# document = loader.load()

# print(document)

# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# chunks = splitter.split_documents(document)

# # 10 chunks & length of chunks
#print(chunks[:10])
#print(len(chunks))

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Create and Save (Old code - Commented Out)
# vectorstore = FAISS.from_documents(chunks, embedding_model)
# vectorstore.save_local("faiss_index")

# Load existing vectorstore from local disk
vectorstore = FAISS.load_local(
    "faiss_index", 
    embedding_model, 
    allow_dangerous_deserialization=True
)

# #print index of first 5 chunks from vectorstore
# #Retrieve document IDs from FAISS docstore
# doc_ids = list(vectorstore.index_to_docstore_id.values())
# print("First 5 Doc IDs:", doc_ids[:5])
# # Print content of 5 documents from vectorstore using docstore search
# for doc_id in doc_ids[:5]:
#     print(vectorstore.docstore.search(doc_id).page_content)
#     print("-" * 50)

# Retriever

retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs={"k": 4})

# print(retriever.invoke("Who is daenerys targaryen?"))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=500)

prompt = PromptTemplate( template="""
You are a helpful assistant that answers questions only based on the provided context.
If the answer is not contained within the context, respond with "I don't know. Please try asking another question."

{context}
Question: {question}""",
input_variables = ["context", "question"]
)

# #test the prompt with a question about Daenerys Targaryen
# response = llm.invoke(prompt.invoke({
#     "context": "Daenerys Targaryen is a fictional character from the 'A Song of Ice and Fire' series by George R.R. Martin.", 
#     "question": "Who is Daenerys?"
# }))

# response2 = llm.invoke(prompt.invoke({
#     "context": "Daenerys Targaryen is a fictional character from the 'A Song of Ice and Fire' series by George R.R. Martin.", 
#     "question": "Who is Amit?"
# }))

# print(response.content)
# print(response2.content)


# Retrieve and combine retrieve document context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

print(main_chain.invoke("Who is Daenerys Targaryen?"))