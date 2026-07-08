import chromadb
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_collection")
collection.add(
    documents=["my name is farah","my name is not farah"],
    metadata=[{"source":"name is true"},{"source":"name is false"}],
    ids=["id1","id2"]
)

result=collection.query(

    query_text=["what is my name?"],
    n_result=1
)

print(results)