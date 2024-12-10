import os
from typing import Annotated, List

from dotenv import load_dotenv
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict


class ConfigManager:
    @staticmethod
    def load_environment():
        load_dotenv()
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in environment")
        return groq_api_key


class ResearchAgentState(TypedDict):
    messages: Annotated[List, add_messages]


class ResearchAgent:
    def __init__(self, model_name: str = "Gemma2-9b-It"):
        """
        Args:
            model_name (str): Name of the language model to use.
        """
        self.api_key = ConfigManager.load_environment()
        self.tools = self._initialize_tools()
        self.llm = self._initialize_llm(model_name)
        self.graph = self._build_workflow()

    def _initialize_tools(self) -> List:
        """Configure research tools."""
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
        wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
        return [
            ArxivQueryRun(api_wrapper=arxiv_wrapper),
            WikipediaQueryRun(api_wrapper=wiki_wrapper)
        ]

    def _initialize_llm(self, model_name: str):
        """Initialize language model with tool binding."""
        llm = ChatGroq(model_name=model_name, groq_api_key=self.api_key)
        return llm.bind_tools(tools=self.tools)

    def _build_workflow(self):
        """Construct the agent's workflow graph."""
        graph_builder = StateGraph(ResearchAgentState)

        graph_builder.add_node("chatbot", self._process_user_input)
        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
        )
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge("chatbot", END)

        return graph_builder.compile(debug=True)

    def _process_user_input(self, state: ResearchAgentState):
        """Process user input and generate response."""
        return {"messages": [self.llm.invoke(state["messages"])]}

    def visualize_workflow(self, output_path: str = "workflow_graph.png"):
        try:
            image_data = self.graph.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as file:
                file.write(image_data)
            print(f"Workflow graph saved to {output_path}")
        except Exception as e:
            print(f"Workflow visualization failed: {e}")

    def interactive_chat(self):
        print("Research Assistant Chat (type 'quit' or 'q' to exit)")
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "q"]:
                print("Goodbye!")
                break

            for event in self.graph.stream({'messages': ("user", user_input)}, stream_mode="values"):
                event["messages"][-1].pretty_print()


def main():
    """Main entry point for the research assistant."""
    try:
        agent = ResearchAgent()
        agent.visualize_workflow()
        agent.interactive_chat()
    except Exception as e:
        print(f"Error initializing research assistant: {e}")


if __name__ == "__main__":
    main()