from flask import Flask,request,jsonify
from graph import final_answer


app=Flask(__name__)

@app.route('/',methods=["POST"])
def send():  
   data=request.json
   answer=final_answer(query=data['query'],querying_tech=data['queryingtech'],retrieving_tech=data['retrievingtech'])
   return jsonify({"response":answer})
if __name__ == '__main__':
   app.run()