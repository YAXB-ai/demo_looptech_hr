import os
from dotenv import load_dotenv
load_dotenv()
from langchain.schema.document import Document
from typing import TypedDict
from langgraph.graph import StateGraph,END
from rag.prompting import general_prompt
from rag.querytranslation import rag_multiquery,rag_decomposition,rag_stepback,rag_hyde,rag_ragfusion
from rag.indexing.Chromadb import rag_chroma_load
from rag.indexing.Raptor import rag_raptor_load
from rag.indexing.colbert import rag_colbert_load




class cook_data(TypedDict):
    query: str
    querying_tech: str
    retrieving_tech:str
    documents:list
    answer:str
    
def retrieve_documents_node(state):
    
    if(state["querying_tech"]=="multiquery"):
        state["documents"]=rag_multiquery(query=state["query"],s=state["retrieving_tech"])
    elif(state["querying_tech"]=="ragfusion"):
        state["documents"]=rag_ragfusion(query=state["query"],s=state["retrieving_tech"])
    elif(state["querying_tech"]=="decomposition"):
        state["documents"]=rag_decomposition(query=state["query"],s=state["retrieving_tech"])
    elif(state["querying_tech"]=="stepback"):
        state["documents"]=rag_stepback(query=state["query"],s=state["retrieving_tech"])
    elif(state["querying_tech"]=="hyde"):
        state["documents"]=rag_hyde(query=state["query"],s=state["retrieving_tech"])
    else:
        if(state["retrieving_tech"]=="chroma"):
            state["documents"]=rag_chroma_load().invoke(state["query"])
        elif(state["retrieving_tech"]=="raptor"):
            state["documents"]=rag_raptor_load().invoke(state["query"])
        else:
            state["documents"]=rag_colbert_load().invoke(state["query"])

    return state
def answer_generation_node(state):
    
    def format_docs(docs):
        page_format=""
        for doc in docs:
            page_format+=(doc.page_content+"\n\n")
        return page_format
    context=format_docs(state["documents"])
    print("context:",context)
    state["answer"]=general_prompt().invoke({"question":state["query"],"context":context})
    return state

graph = StateGraph(cook_data)
graph.add_node("retrieve_documents",retrieve_documents_node)
graph.add_node("answer_generation_node",answer_generation_node)
graph.set_entry_point("retrieve_documents")
graph.add_edge("retrieve_documents","answer_generation_node")
graph.add_edge("answer_generation_node",END)
apps=graph.compile()



def final_answer(query,querying_tech,retrieving_tech):
    final_answer=apps.invoke({"query":query,"querying_tech":querying_tech,"retrieving_tech":retrieving_tech})

    return final_answer["answer"]


# from rag.docloaders import rag_pdf_loader
# from rag.indexing.colbert import rag_colbert_save,rag_colbert_load
# from rag.indexing.Chromadb import rag_chroma_save,rag_chroma_load
# # print(rag_chroma_load().invoke("tell me about looptech hr"))

# pages=rag_pdf_loader("./assets/C_T_AIC2_REV0.PDF.pdf")
# format_data=""
# print(len(pages))
# for i,page in enumerate(pages,start=0):
#     print("****",i,"****","\n")
#     format_data+=page.page_content
#     # print(page)
#     # rag_chroma_save(page)


# print(format_data)   


# data=Document(page_content=format_data,metadata=pages[0].metadata)

# print(data)
# # print(pages[182])
# print(type(data))
# rag_chroma_save(data)