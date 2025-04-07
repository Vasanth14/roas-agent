from flask import Flask, request, render_template, session
import pandas as pd
import json
import os

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama

app = Flask(__name__)
app.secret_key = os.urandom(24)  # for session management

global_df = pd.DataFrame()

# LLM setup
llm = Ollama(model="llama2")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Tool to analyze Pandas data
def analyze_data(query: str) -> str:
    global global_df
    try:
        if global_df.empty:
            return "No data loaded yet."

        if "CTR" in query or "click-through" in query.lower():
            global_df["CTR"] = (global_df["Clicks"] / global_df["Impressions"]) * 100
            top = global_df.sort_values(by="CTR", ascending=False).head(3)
            return f"Top campaigns by CTR (%):\n{top[['Campaign', 'CTR']].to_string(index=False)}"

        elif "conversion" in query.lower():
            global_df["ConversionRate"] = (
                global_df["Conversions"] / global_df["Clicks"]
            ) * 100
            top = global_df.sort_values(by="ConversionRate", ascending=False).head(3)
            return f"Top campaigns by Conversion Rate (%):\n{top[['Campaign', 'ConversionRate']].to_string(index=False)}"

        return "Please ask about CTR or conversion performance."

    except Exception as e:
        return f"Error: {str(e)}"


tools = [
    Tool(
        name="CampaignDataAnalyzer",
        func=analyze_data,
        description=(
            "Use this tool to answer any query that involves calculations or metrics from the campaign CSV, "
            "such as total clicks, impressions, CTR, or conversion rates."
        )
    )
]


agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/upload", methods=["POST"])
def upload():
    global global_df
    file = request.files["file"]
    if file.filename == "":
        return render_template("chat.html", error="Please upload a CSV file.")

    try:
        global_df = pd.read_csv(file)
        session["csv_loaded"] = True
        return render_template(
            "chat.html",
            success="CSV uploaded. Ask me anything about the campaign performance!",
        )
    except Exception as e:
        return render_template("chat.html", error=f"Error reading CSV: {str(e)}")


@app.route("/chat", methods=["POST"])
def chat():
    if not session.get("csv_loaded"):
        return render_template("chat.html", error="Please upload a CSV first.")

    user_input = request.form.get("message")
    if not user_input:
        return render_template("chat.html", error="Please enter a message.")

    # Prompt for JSON-only output
    query = f"""
    You are a data analysis assistant for marketing campaigns. Use the available tools to calculate or summarize any numerical data such as Budget, Bids, Targeting etc.

    If you get a query that requires numeric insights, use the tools. Then respond **only** with a JSON array of actionable recommendations or insights.

    Query: {user_input}

    Format:
    recommendations = [
        {{"recommendation": "..."}},
        {{"recommendation": "..."}},
        {{"recommendation": "..."}}
    ]
    """



    try:
        output = llm.invoke(query)
        start = output.find("[")
        end = output.rfind("]") + 1
        json_array = json.loads(output[start:end])
        recommendations = json_array
    except Exception as e:
        return render_template(
            "chat.html",
            user_input=user_input,
            response=f"Could not parse LLM output.\n\nRaw Output:\n{output}",
            chat_history=memory.buffer,
        )


    return render_template(
        "chat.html",
        user_input=user_input,
        recommendations=recommendations,
        chat_history=memory.buffer,
    )



if __name__ == "__main__":
    app.run(debug=True)
