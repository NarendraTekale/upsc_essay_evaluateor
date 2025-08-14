from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import certifi
import os
import operator

app = Flask(__name__)

# Load environment variables
load_dotenv()
os.environ['SSL_CERT_FILE'] = certifi.where()

# Initialize the model
model = ChatOpenAI(model='gpt-4')

class EvaluationSchema(BaseModel):
    feedback: str = Field(description='Detailed feedback for the essay')
    score: float = Field(description='Score out of 10', ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[float], operator.add]
    avg_score: float

def evaluate_language(state: UPSCState):
    prompt = f'Evaluate the language quality of the following essay and provide feedback and assign a score out of 10 \\n {state["essay"]}'
    output = structured_model.invoke(prompt)
    return {'language_feedback': output.feedback, 'individual_scores': [output.score]}

def evaluate_analysis(state: UPSCState):
    prompt = f'Evaluate the depth of analysis of the following essay and provide feedback and assign a score out of 10 \\n {state["essay"]}'
    output = structured_model.invoke(prompt)
    return {'analysis_feedback': output.feedback, 'individual_scores': [output.score]}

def evaluate_thought(state: UPSCState):
    prompt = f'Evaluate the clarity of thought of the following essay and provide feedback and assign a score out of 10 \\n {state["essay"]}'
    output = structured_model.invoke(prompt)
    return {'clarity_feedback': output.feedback, 'individual_scores': [output.score]}

def final_evaluation(state: UPSCState):
    # summary feedback
    prompt = f'Based on the following feedbacks create a summarized feedback \\n language feedback - {state["language_feedback"]} \\n depth of analysis feedback - {state["analysis_feedback"]} \\n clarity of thought feedback - {state["clarity_feedback"]}'
    overall_feedback = model.invoke(prompt).content
    
    # avg calculate
    avg_score = sum(state['individual_scores'])/len(state['individual_scores'])
    
    return {'overall_feedback': overall_feedback, 'avg_score': avg_score}

# Create the workflow
graph = StateGraph(UPSCState)


graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_thought', evaluate_thought)
graph.add_node('final_evaluation', final_evaluation)

# Add edges
graph = StateGraph(UPSCState)

graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_thought', evaluate_thought)
graph.add_node('final_evaluation', final_evaluation)

# edges
graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_thought')

graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_thought', 'final_evaluation')

graph.add_edge('final_evaluation', END)

workflow = graph.compile()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        essay = request.form['essay']
        
        if not essay.strip():
            return render_template('index.html', error="Please enter an essay to evaluate.")
        
        initial_state = {
            'essay': essay
        }
        
        try:
            result = workflow.invoke(initial_state)
            return render_template('index.html', result=result, essay=essay)
        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}")
    
    return render_template('index.html')

def health_check():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
